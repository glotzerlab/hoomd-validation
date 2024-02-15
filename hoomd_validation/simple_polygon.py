# Copyright (c) 2022-2023 The Regents of the University of Michigan.
# Part of HOOMD-blue, released under the BSD 3-Clause License.

"""Simple polygon equation of state validation test."""

import json
import os
import pathlib

import util
from config import CONFIG
from flow import aggregator
from project_class import Project

# Run parameters shared between simulations.
# Step counts must be even and a multiple of the log quantity period.
RANDOMIZE_STEPS = 20_000
EQUILIBRATE_STEPS = 100_000
RUN_STEPS = 500_000
RESTART_STEPS = RUN_STEPS // 50
TOTAL_STEPS = RANDOMIZE_STEPS + EQUILIBRATE_STEPS + RUN_STEPS
SHAPE_VERTICES = [
    (-0.5, 0.5),
    (-0.5, -0.5),
    (0.5, -0.5),
    (0.5, -0.25),
    (-0.25, -0.25),
    (-0.25, 0.25),
    (0.5, 0.25),
    (0.5, 0.5),
]

WRITE_PERIOD = 1_000
LOG_PERIOD = {'trajectory': 50_000, 'quantities': 100}
NUM_CPU_RANKS = min(8, CONFIG['max_cores_sim'])

WALLTIME_STOP_SECONDS = CONFIG['max_walltime'] * 3600 - 10 * 60


def job_statepoints():
    """list(dict): A list of statepoints for this subproject."""
    num_particles = 64**2
    replicate_indices = range(CONFIG['replicates'])
    # statepoint chosen from initial tests: mean +- std_err_means =
    # 6.205344838518146 +- 0.0011239315686705335
    # based on nvt sims at density = 0.75 over 32 replicas
    params_list = [(0.75, 6.205)]  # mean +- st
    for density, pressure in params_list:
        for idx in replicate_indices:
            yield (
                {
                    'subproject': 'simple_polygon',
                    'density': density,
                    'pressure': pressure,
                    'num_particles': num_particles,
                    'replicate_idx': idx,
                }
            )


def is_simple_polygon(job):
    """Test if a given job is part of the simple_polygon subproject."""
    return job.cached_statepoint['subproject'] == 'simple_polygon'


partition_jobs_cpu_serial = aggregator.groupsof(
    num=min(CONFIG['replicates'], CONFIG['max_cores_submission']),
    sort_by='density',
    select=is_simple_polygon,
)

partition_jobs_cpu_mpi = aggregator.groupsof(
    num=min(CONFIG['replicates'], CONFIG['max_cores_submission'] // NUM_CPU_RANKS),
    sort_by='density',
    select=is_simple_polygon,
)


@Project.post.isfile('simple_polygon_initial_state.gsd')
@Project.operation(
    directives=dict(
        executable=CONFIG['executable'],
        nranks=util.total_ranks_function(NUM_CPU_RANKS),
        walltime=1,
    ),
    aggregator=partition_jobs_cpu_mpi,
)
def simple_polygon_create_initial_state(*jobs):
    """Create initial system configuration."""
    import itertools

    import hoomd
    import numpy

    communicator = hoomd.communicator.Communicator(ranks_per_partition=NUM_CPU_RANKS)
    job = jobs[communicator.partition]

    if communicator.rank == 0:
        print('starting simple_polygon_create_initial_state:', job)

    num_particles = job.cached_statepoint['num_particles']
    density = job.cached_statepoint['density']

    box_volume = num_particles / density
    L = box_volume ** (1 / 2.0)

    N = int(numpy.ceil(num_particles ** (1.0 / 2.0)))
    x = numpy.linspace(-L / 2, L / 2, N, endpoint=False)

    particle_spacing = 1.0
    if x[1] - x[0] < particle_spacing:
        raise RuntimeError('density too high to initialize on square lattice')

    position_2d = list(itertools.product(x, repeat=2))[:num_particles]

    # create snapshot
    device = hoomd.device.CPU(
        communicator=communicator,
        message_filename=util.get_message_filename(job, 'create_initial_state.log'),
    )
    snap = hoomd.Snapshot(communicator)

    if communicator.rank == 0:
        snap.particles.N = num_particles
        snap.particles.types = ['A']
        snap.configuration.box = [L, L, 0, 0, 0, 0]
        snap.particles.position[:, 0:2] = position_2d
        snap.particles.typeid[:] = [0] * num_particles

    # Use hard sphere Monte-Carlo to randomize the initial configuration
    mc = hoomd.hpmc.integrate.SimplePolygon(default_d=0.05, default_a=0.1)
    mc.shape['A'] = dict(vertices=SHAPE_VERTICES)

    sim = hoomd.Simulation(device=device, seed=util.make_seed(job))
    sim.create_state_from_snapshot(snap)
    sim.operations.integrator = mc

    device.notice('Randomizing initial state...')
    sim.run(RANDOMIZE_STEPS)
    device.notice(f'Move counts: {mc.translate_moves} {mc.rotate_moves}')
    device.notice('Done.')

    trajectory_logger = hoomd.logging.Logger(categories=['object'])
    trajectory_logger.add(mc, quantities=['type_shapes'])

    hoomd.write.GSD.write(
        state=sim.state,
        filename=job.fn('simple_polygon_initial_state.gsd'),
        mode='wb',
        logger=trajectory_logger,
    )

    if communicator.rank == 0:
        print(f'completed simple_polygon_create_initial_state: {job}')


def make_mc_simulation(job, device, initial_state, sim_mode, extra_loggables=None):
    """Make a simple polygon MC Simulation.

    Args:
        job (`signac.job.Job`): Signac job object.

        device (`hoomd.device.Device`): Device object.

        initial_state (str): Path to the gsd file to be used as an initial state
            for the simulation.

        sim_mode (str): String defining the simulation mode.

        extra_loggables (list[tuple]): List of extra loggables to log to gsd
            files. Each tuple is a pair of the instance and the loggable
            quantity name.
    """
    import hoomd
    import numpy
    from custom_actions import ComputeDensity

    if extra_loggables is None:
        extra_loggables = []

    # integrator
    mc = hoomd.hpmc.integrate.SimplePolygon(default_d=0.05, default_a=0.1)
    mc.shape['A'] = dict(vertices=SHAPE_VERTICES)

    # compute the density and pressure
    compute_density = ComputeDensity()
    sdf = hoomd.hpmc.compute.SDF(xmax=0.02, dx=1e-4)

    logger = hoomd.logging.Logger(categories=['scalar', 'sequence'])
    logger.add(mc, quantities=['translate_moves'])
    logger.add(sdf, quantities=['betaP'])
    logger.add(compute_density, quantities=['density'])
    for loggable, quantity in extra_loggables:
        logger.add(loggable, quantities=[quantity])

    trajectory_logger = hoomd.logging.Logger(categories=['object'])
    trajectory_logger.add(mc, quantities=['type_shapes'])

    # make simulation
    sim = util.make_simulation(
        job=job,
        device=device,
        initial_state=initial_state,
        integrator=mc,
        sim_mode=sim_mode,
        logger=logger,
        table_write_period=WRITE_PERIOD,
        trajectory_write_period=LOG_PERIOD['trajectory'],
        log_write_period=LOG_PERIOD['quantities'],
        log_start_step=RANDOMIZE_STEPS + EQUILIBRATE_STEPS,
        trajectory_logger=trajectory_logger,
    )

    sim.operations.computes.append(sdf)
    compute_density.attach(sim)

    for loggable, _ in extra_loggables:
        # call attach method explicitly so we can access simulation state when
        # computing the loggable quantity
        if hasattr(loggable, 'attach'):
            loggable.attach(sim)

    # move size tuner
    move_size_tuner = hoomd.hpmc.tune.MoveSize.scale_solver(
        moves=['d', 'a'],
        target=0.2,
        max_translation_move=0.5,
        max_rotation_move=2 * numpy.pi / 4,
        trigger=hoomd.trigger.And(
            [
                hoomd.trigger.Periodic(100),
                hoomd.trigger.Before(RANDOMIZE_STEPS + EQUILIBRATE_STEPS // 2),
            ]
        ),
    )
    sim.operations.add(move_size_tuner)

    return sim


def run_nvt_sim(job, device, complete_filename):
    """Run MC sim in NVT."""
    import hoomd

    sim_mode = 'nvt'
    restart_filename = util.get_job_filename(sim_mode, device, 'restart', 'gsd')
    if job.isfile(restart_filename):
        initial_state = job.fn(restart_filename)
        restart = True
    else:
        initial_state = job.fn('simple_polygon_initial_state.gsd')
        restart = False

    sim = make_mc_simulation(job, device, initial_state, sim_mode, extra_loggables=[])

    if not restart:
        # equilibrate
        device.notice('Equilibrating...')
        sim.run(EQUILIBRATE_STEPS // 2)
        sim.run(EQUILIBRATE_STEPS // 2)
        device.notice('Done.')

        # Print acceptance ratio as measured during the 2nd half of the
        # equilibration.
        translate_moves = sim.operations.integrator.translate_moves
        translate_acceptance = translate_moves[0] / sum(translate_moves)
        rotate_moves = sim.operations.integrator.rotate_moves
        rotate_acceptance = rotate_moves[0] / sum(rotate_moves)
        device.notice(f'Translate move acceptance: {translate_acceptance}')
        device.notice(f'Trial translate move size: {sim.operations.integrator.d["A"]}')
        device.notice(f'Rotate move acceptance: {rotate_acceptance}')
        device.notice(f'Trial rotate move size: {sim.operations.integrator.a["A"]}')

        # save move size to a file
        if device.communicator.rank == 0:
            name = util.get_job_filename(sim_mode, device, 'move_size', 'json')
            with open(job.fn(name), 'w') as f:
                json.dump(
                    dict(
                        d_A=sim.operations.integrator.d['A'],
                        a_A=sim.operations.integrator.a['A'],
                    ),
                    f,
                )
    else:
        device.notice('Restarting...')
        # read move size from the file
        name = util.get_job_filename(sim_mode, device, 'move_size', 'json')
        with open(job.fn(name)) as f:
            data = json.load(f)

        sim.operations.integrator.d['A'] = data['d_A']
        sim.operations.integrator.a['A'] = data['a_A']
        mcd = sim.operations.integrator.d['A']
        device.notice(f'Restored translate trial move size: {mcd}')
        mca = sim.operations.integrator.a['A']
        device.notice(f'Restored rotate trial move size: {mca}')

    # run
    device.notice('Running...')

    util.run_up_to_walltime(
        sim=sim,
        end_step=TOTAL_STEPS,
        steps=RESTART_STEPS,
        walltime_stop=WALLTIME_STOP_SECONDS,
    )

    hoomd.write.GSD.write(state=sim.state, filename=job.fn(restart_filename), mode='wb')

    if sim.timestep == TOTAL_STEPS:
        pathlib.Path(job.fn(complete_filename)).touch()
        device.notice('Done.')
    else:
        device.notice(
            'Ending run early due to walltime limits at:'
            f'{device.communicator.walltime}'
        )


def run_npt_sim(job, device, complete_filename):
    """Run MC sim in NPT."""
    import hoomd

    # device
    sim_mode = 'npt'
    restart_filename = util.get_job_filename(sim_mode, device, 'restart', 'gsd')
    if job.isfile(restart_filename):
        initial_state = job.fn(restart_filename)
        restart = True
    else:
        initial_state = job.fn('simple_polygon_initial_state.gsd')
        restart = False

    # box updates
    boxmc = hoomd.hpmc.update.BoxMC(
        betaP=job.cached_statepoint['pressure'], trigger=hoomd.trigger.Periodic(1)
    )
    boxmc.volume = dict(weight=1.0, mode='ln', delta=1e-6)

    # simulation
    sim = make_mc_simulation(
        job,
        device,
        initial_state,
        sim_mode,
        extra_loggables=[
            (boxmc, 'volume_moves'),
        ],
    )

    sim.operations.add(boxmc)

    boxmc_tuner = hoomd.hpmc.tune.BoxMCMoveSize.scale_solver(
        trigger=hoomd.trigger.And(
            [
                hoomd.trigger.Periodic(400),
                hoomd.trigger.Before(RANDOMIZE_STEPS + EQUILIBRATE_STEPS // 2),
            ]
        ),
        boxmc=boxmc,
        moves=['volume'],
        target=0.5,
    )
    sim.operations.add(boxmc_tuner)

    if not restart:
        # equilibrate
        device.notice('Equilibrating...')
        sim.run(EQUILIBRATE_STEPS // 2)
        sim.run(EQUILIBRATE_STEPS // 2)
        device.notice('Done.')

        # Print acceptance ratio as measured during the 2nd half of the
        # equilibration.
        translate_moves = sim.operations.integrator.translate_moves
        translate_acceptance = translate_moves[0] / sum(translate_moves)
        device.notice(f'Translate move acceptance: {translate_acceptance}')
        device.notice(f'Translate trial move size: {sim.operations.integrator.d["A"]}')
        rotate_moves = sim.operations.integrator.rotate_moves
        rotate_acceptance = rotate_moves[0] / sum(rotate_moves)
        device.notice(f'Rotate move acceptance: {rotate_acceptance}')
        device.notice(f'Rotate trial move size: {sim.operations.integrator.a["A"]}')

        volume_moves = boxmc.volume_moves
        volume_acceptance = volume_moves[0] / sum(volume_moves)
        device.notice(f'Volume move acceptance: {volume_acceptance}')
        device.notice(f'Volume move size: {boxmc.volume["delta"]}')

        # save move sizes to a file
        if device.communicator.rank == 0:
            name = util.get_job_filename(sim_mode, device, 'move_size', 'json')
            with open(job.fn(name), 'w') as f:
                json.dump(
                    dict(
                        d_A=sim.operations.integrator.d['A'],
                        a_A=sim.operations.integrator.a['A'],
                        volume_delta=boxmc.volume['delta'],
                    ),
                    f,
                )
    else:
        device.notice('Restarting...')
        # read move size from the file
        name = util.get_job_filename(sim_mode, device, 'move_size', 'json')
        with open(job.fn(name)) as f:
            data = json.load(f)

        sim.operations.integrator.d['A'] = data['d_A']
        sim.operations.integrator.a['A'] = data['a_A']
        mcd = sim.operations.integrator.d['A']
        device.notice(f'Restored translate trial move size: {mcd}')
        mca = sim.operations.integrator.a['A']
        device.notice(f'Restored rotate trial move size: {mca}')
        boxmc.volume = dict(weight=1.0, mode='ln', delta=data['volume_delta'])
        device.notice(f'Restored volume move size: {boxmc.volume["delta"]}')

    # run
    device.notice('Running...')
    util.run_up_to_walltime(
        sim=sim,
        end_step=TOTAL_STEPS,
        steps=100_000,
        walltime_stop=WALLTIME_STOP_SECONDS,
    )

    hoomd.write.GSD.write(state=sim.state, filename=job.fn(restart_filename), mode='wb')

    if sim.timestep == TOTAL_STEPS:
        pathlib.Path(job.fn(complete_filename)).touch()
        device.notice('Done.')
    else:
        device.notice(
            'Ending run early due to walltime limits at:'
            f'{device.communicator.walltime}'
        )


sampling_jobs = []
job_definitions = [
    {
        'mode': 'nvt',
        'device_name': 'cpu',
        'ranks_per_partition': NUM_CPU_RANKS,
        'aggregator': partition_jobs_cpu_mpi,
    },
    {
        'mode': 'npt',
        'device_name': 'cpu',
        'ranks_per_partition': NUM_CPU_RANKS,
        'aggregator': partition_jobs_cpu_mpi,
    },
]


def add_sampling_job(mode, device_name, ranks_per_partition, aggregator):
    """Add a sampling job to the workflow."""
    directives = dict(
        walltime=CONFIG['max_walltime'],
        executable=CONFIG['executable'],
        nranks=util.total_ranks_function(ranks_per_partition),
    )

    @Project.pre.after(simple_polygon_create_initial_state)
    @Project.post.isfile(f'{mode}_{device_name}_complete')
    @Project.operation(
        name=f'simple_polygon_{mode}_{device_name}',
        directives=directives,
        aggregator=aggregator,
    )
    def sampling_operation(*jobs):
        """Perform sampling simulation given the definition."""
        import hoomd

        communicator = hoomd.communicator.Communicator(
            ranks_per_partition=ranks_per_partition
        )
        job = jobs[communicator.partition]

        if communicator.rank == 0:
            print(f'starting simple_polygon_{mode}_{device_name}:', job)

        device = hoomd.device.CPU(
            communicator=communicator,
            message_filename=util.get_message_filename(
                job, f'{mode}_{device_name}.log'
            ),
        )

        globals().get(f'run_{mode}_sim')(
            job, device, complete_filename=f'{mode}_{device_name}_complete'
        )

        if communicator.rank == 0:
            print(f'completed simple_polygon_{mode}_{device_name} {job}')

    sampling_jobs.append(sampling_operation)


for definition in job_definitions:
    add_sampling_job(**definition)


@Project.pre(is_simple_polygon)
@Project.pre.after(*sampling_jobs)
@Project.post.true('simple_polygon_analysis_complete')
@Project.operation(
    directives=dict(walltime=CONFIG['short_walltime'], executable=CONFIG['executable'])
)
def simple_polygon_analyze(job):
    """Analyze the output of all simulation modes."""
    import matplotlib
    import matplotlib.figure
    import matplotlib.style
    import numpy

    matplotlib.style.use('fivethirtyeight')

    print('starting simple_polygon_analyze:', job)

    sim_modes = []
    for _ensemble in ['nvt', 'npt']:
        if job.isfile(f'{_ensemble}_cpu_quantities.h5'):
            sim_modes.append(f'{_ensemble}_cpu')

    util._sort_sim_modes(sim_modes)

    timesteps = {}
    pressures = {}
    densities = {}

    for sim_mode in sim_modes:
        log_traj = util.read_log(job.fn(sim_mode + '_quantities.h5'))

        timesteps[sim_mode] = log_traj['hoomd-data/Simulation/timestep']

        pressures[sim_mode] = log_traj['hoomd-data/hpmc/compute/SDF/betaP']

        densities[sim_mode] = log_traj[
            'hoomd-data/custom_actions/ComputeDensity/density'
        ]

    # save averages
    for mode in sim_modes:
        job.document[mode] = dict(
            pressure=float(numpy.mean(pressures[mode])),
            density=float(numpy.mean(densities[mode])),
        )

    # Plot results
    fig = matplotlib.figure.Figure(figsize=(10, 10 / 1.618 * 2), layout='tight')
    ax = fig.add_subplot(2, 1, 1)
    util.plot_timeseries(
        ax=ax,
        timesteps=timesteps,
        data=densities,
        ylabel=r'$\rho$',
        expected=job.cached_statepoint['density'],
        max_points=500,
    )
    ax.legend()

    ax = fig.add_subplot(2, 1, 2)
    util.plot_timeseries(
        ax=ax,
        timesteps=timesteps,
        data=pressures,
        ylabel=r'$\beta P$',
        expected=job.cached_statepoint['pressure'],
        max_points=500,
    )

    fig.suptitle(
        f'$\\rho={job.cached_statepoint['density']}$, '
        f'$N={job.cached_statepoint['num_particles']}$, '
        f'replicate={job.cached_statepoint['replicate_idx']}'
    )
    fig.savefig(job.fn('nvt_npt_plots.svg'), bbox_inches='tight')

    job.document['simple_polygon_analysis_complete'] = True


@Project.pre(lambda *jobs: util.true_all(*jobs, key='simple_polygon_analysis_complete'))
@Project.post(
    lambda *jobs: util.true_all(*jobs, key='simple_polygon_compare_modes_complete')
)
@Project.operation(
    directives=dict(executable=CONFIG['executable']),
    aggregator=aggregator.groupby(
        key=['density', 'num_particles'],
        sort_by='replicate_idx',
        select=is_simple_polygon,
    ),
)
def simple_polygon_compare_modes(*jobs):
    """Compares the tested simulation modes."""
    import matplotlib
    import matplotlib.figure
    import matplotlib.style
    import numpy

    matplotlib.style.use('fivethirtyeight')

    print('starting simple_polygon_compare_modes:', jobs[0])

    sim_modes = []
    for _ensemble in ['nvt', 'npt']:
        if jobs[0].isfile(f'{_ensemble}_cpu_quantities.h5'):
            sim_modes.append(f'{_ensemble}_cpu')

    util._sort_sim_modes(sim_modes)

    quantity_names = ['density', 'pressure']

    labels = {
        'density': r'$\frac{\rho_\mathrm{sample} - \rho}{\rho} \cdot 1000$',
        'pressure': r'$\frac{P_\mathrm{sample} - P}{P} \cdot 1000$',
    }

    # grab the common statepoint parameters
    set_density = jobs[0].sp.density
    set_pressure = jobs[0].sp.pressure
    num_particles = jobs[0].sp.num_particles

    quantity_reference = dict(density=set_density, pressure=set_pressure)

    fig = matplotlib.figure.Figure(figsize=(10, 10 / 1.618 * 2), layout='tight')
    fig.suptitle(f'$\\rho={set_density}$, $N={num_particles}$')

    for i, quantity_name in enumerate(quantity_names):
        ax = fig.add_subplot(2, 1, i + 1)

        # organize data from jobs
        quantities = {mode: [] for mode in sim_modes}
        for jb in jobs:
            for mode in sim_modes:
                quantities[mode].append(getattr(getattr(jb.doc, mode), quantity_name))

        if quantity_reference[quantity_name] is not None:
            reference = quantity_reference[quantity_name]
        else:
            avg_value = {mode: numpy.mean(quantities[mode]) for mode in sim_modes}
            reference = numpy.mean([avg_value[mode] for mode in sim_modes])

        avg_quantity, stderr_quantity = util.plot_vs_expected(
            ax=ax,
            values=quantities,
            ylabel=labels[quantity_name],
            expected=reference,
            relative_scale=1000,
            separate_nvt_npt=True,
        )

    filename = f'simple_polygon_compare_density{round(set_density, 2)}.svg'
    fig.savefig(os.path.join(jobs[0]._project.path, filename), bbox_inches='tight')

    for job in jobs:
        job.document['simple_polygon_compare_modes_complete'] = True
