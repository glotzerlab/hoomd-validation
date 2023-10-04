# Copyright (c) 2022-2023 The Regents of the University of Michigan.
# Part of HOOMD-blue, released under the BSD 3-Clause License.

"""Test for consistency between NVT and NPT simulations of patchy particles."""

from config import CONFIG
from project_class import Project
from flow import aggregator
import util
import os
import json
import pathlib

# Run parameters shared between simulations.
# Step counts must be even and a multiple of the log quantity period.
RANDOMIZE_STEPS = 20_000
EQUILIBRATE_STEPS = 100_000
RUN_STEPS = 1_000_000
RESTART_STEPS = RUN_STEPS // 50
TOTAL_STEPS = RANDOMIZE_STEPS + EQUILIBRATE_STEPS + RUN_STEPS

WRITE_PERIOD = 1_000
LOG_PERIOD = {'trajectory': 50_000, 'quantities': 500}
NUM_CPU_RANKS = min(9, CONFIG["max_cores_sim"])

WALLTIME_STOP_SECONDS = CONFIG["max_walltime"] * 3600 - 10 * 60


def job_statepoints():
    """list(dict): A list of statepoints for this subproject."""
    num_particles = 8**3
    replicate_indices = range(CONFIG["replicates"])
    # statepoint chosen to be in a dense liquid
    # nvt simulations at density = 0.95 yielded a measured pressure of
    # 6.415 +/- 0.003 (mean +/- std. error of means) over 32 replicas
    params_list = [
        (1000.0, 0.95, 12.12, 0.7, 1.5),
        (10.0, 0.95, 11.83, 0.7, 1.5),
        (1.0, 0.95, 8.27, 0.7, 1.5),
        # next 3 are from 10.1063/1.3054361
        # pressure from NVT simulations
        (0.5714, 0.8, -5.871736803897682, 1.0, 1.5),
        (1.0, 0.8, -1.0235450460184001, 1.0, 1.5),
        (3.0, 0.8, 3.7415391338219885, 1.0, 1.5),
        # next 2 are from 10.1063/1.3054361
        # pressure from webplotdigitizer fig 7
        (1.0, 0.8, 1.70178459919393820, 1.0, 1.5
         ),  # pressure = pressure = pressure/kT
        (3.0, 0.8, 13.549010655204555, 1.0, 1.5),  # pressure = pressure
        (3.0, 0.8, 4.516336885068185, 1.0, 1.5),  # pressure = pressure/kT
    ]  # kT, rho, pressure, chi, lambda_
    for temperature, density, pressure, chi, lambda_ in params_list:
        for idx in replicate_indices:
            yield ({
                "subproject": "patchy_particle_pressure",
                "density": density,
                "pressure": pressure,
                "chi": chi,
                "lambda_": lambda_,
                "num_particles": num_particles,
                "replicate_idx": idx,
                "temperature": temperature,
            })


def is_patchy_particle_pressure(job):
    """Test if a job is part of the patchy_particle_pressure subproject."""
    return job.statepoint['subproject'] == 'patchy_particle_pressure'


def is_patchy_particle_pressure_positive_pressure(job):
    """Test if a job is part of the patchy_particle_pressure subproject."""
    return job.statepoint[
        'subproject'] == 'patchy_particle_pressure' and job.statepoint[
            'pressure'] > 0.0


partition_jobs_cpu_serial = aggregator.groupsof(
    num=min(CONFIG["replicates"], CONFIG["max_cores_submission"]),
    sort_by='density',
    select=is_patchy_particle_pressure,
)

partition_jobs_cpu_mpi_nvt = aggregator.groupsof(
    num=min(CONFIG["replicates"],
            CONFIG["max_cores_submission"] // NUM_CPU_RANKS),
    sort_by='density',
    select=is_patchy_particle_pressure,
)

partition_jobs_cpu_mpi_npt = aggregator.groupsof(
    num=min(CONFIG["replicates"],
            CONFIG["max_cores_submission"] // NUM_CPU_RANKS),
    sort_by='density',
    select=is_patchy_particle_pressure_positive_pressure,
)

partition_jobs_gpu = aggregator.groupsof(
    num=min(CONFIG["replicates"], CONFIG["max_gpus_submission"]),
    sort_by='density',
    select=is_patchy_particle_pressure,
)


@Project.post.isfile('patchy_particle_pressure_initial_state.gsd')
@Project.operation(
    directives=dict(executable=CONFIG["executable"],
                    nranks=util.total_ranks_function(NUM_CPU_RANKS),
                    walltime=1),
    aggregator=partition_jobs_cpu_mpi_nvt,
)
def patchy_particle_pressure_create_initial_state(*jobs):
    """Create initial system configuration."""
    import hoomd
    import numpy
    import itertools

    communicator = hoomd.communicator.Communicator(
        ranks_per_partition=NUM_CPU_RANKS)
    job = jobs[communicator.partition]

    if communicator.rank == 0:
        print('starting patchy_particle_pressure_create_initial_state:', job)

    num_particles = job.statepoint['num_particles']
    density = job.statepoint['density']
    temperature = job.statepoint['temperature']
    chi = job.statepoint['chi']
    lambda_ = job.statepoint['lambda_']

    box_volume = num_particles / density
    L = box_volume**(1 / 3.)

    N = int(numpy.ceil(num_particles**(1. / 3.)))
    x = numpy.linspace(-L / 2, L / 2, N, endpoint=False)

    if x[1] - x[0] < 1.0:
        raise RuntimeError('density too high to initialize on square lattice')

    position = list(itertools.product(x, repeat=3))[:num_particles]

    # create snapshot
    device = hoomd.device.CPU(
        communicator=communicator,
        message_filename=job.fn('create_initial_state.log'))
    snap = hoomd.Snapshot(communicator)

    if communicator.rank == 0:
        snap.particles.N = num_particles
        snap.particles.types = ['A']
        snap.configuration.box = [L, L, L, 0, 0, 0]
        snap.particles.position[:] = position
        snap.particles.typeid[:] = [0] * num_particles

    # Use hard sphere + patches Monte Carlo to randomize the initial
    # configuration
    mc = hoomd.hpmc.integrate.Sphere(default_d=0.05, default_a=0.1)
    diameter = 1.0
    mc.shape['A'] = dict(diameter=diameter, orientable=True)
    delta = 2 * numpy.arcsin(numpy.sqrt(chi))
    patch_code = util._single_patch_kern_frenkel_code(delta, lambda_, diameter,
                                                      temperature)
    r_cut = diameter + diameter * (lambda_ - 1)
    patches = hoomd.hpmc.pair.user.CPPPotential(r_cut=r_cut,
                                                code=patch_code,
                                                param_array=[])
    mc.pair_potential = patches

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
        filename=job.fn("patchy_particle_pressure_initial_state.gsd"),
        mode='wb',
        logger=trajectory_logger,
    )

    if communicator.rank == 0:
        print(f'completed patchy_particle_pressure_create_initial_state: '
              f'{job} in {communicator.walltime} s')


def make_mc_simulation(job,
                       device,
                       initial_state,
                       sim_mode,
                       extra_loggables=[]):
    """Make a patchy sphere MC Simulation.

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

    # integrator and patchy potential
    temperature = job.statepoint['temperature']
    chi = job.statepoint['chi']
    lambda_ = job.statepoint['lambda_']
    diameter = 1.0
    mc = hoomd.hpmc.integrate.Sphere(default_d=0.05, default_a=0.1)
    mc.shape['A'] = dict(diameter=diameter, orientable=True)
    delta = 2 * numpy.arcsin(numpy.sqrt(chi))
    patch_code = util._single_patch_kern_frenkel_code(delta, lambda_, diameter,
                                                      temperature)
    r_cut = diameter + diameter * (lambda_ - 1)
    patches = hoomd.hpmc.pair.user.CPPPotential(r_cut=r_cut,
                                                code=patch_code,
                                                param_array=[])
    mc.pair_potential = patches

    # compute the density and pressure
    compute_density = ComputeDensity()
    sdf = hoomd.hpmc.compute.SDF(xmax=0.02, dx=1e-4)

    # log to gsd
    logger_gsd = hoomd.logging.Logger(categories=['scalar', 'sequence'])
    logger_gsd.add(mc, quantities=['translate_moves'])
    logger_gsd.add(sdf, quantities=['betaP'])
    logger_gsd.add(compute_density, quantities=['density'])
    logger_gsd.add(patches, quantities=['energy'])
    for loggable, quantity in extra_loggables:
        logger_gsd.add(loggable, quantities=[quantity])

    trajectory_logger = hoomd.logging.Logger(categories=['object'])
    trajectory_logger.add(mc, quantities=['type_shapes'])

    # make simulation
    sim = util.make_simulation(
        job=job,
        device=device,
        initial_state=initial_state,
        integrator=mc,
        sim_mode=sim_mode,
        logger=logger_gsd,
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
        trigger=hoomd.trigger.And([
            hoomd.trigger.Periodic(100),
            hoomd.trigger.Before(RANDOMIZE_STEPS + EQUILIBRATE_STEPS // 2)
        ]))
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
        initial_state = job.fn('patchy_particle_pressure_initial_state.gsd')
        restart = False

    sim = make_mc_simulation(job,
                             device,
                             initial_state,
                             sim_mode,
                             extra_loggables=[])

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
        device.notice(
            f'Trial translate move size: {sim.operations.integrator.d["A"]}')
        device.notice(f'Rotate move acceptance: {rotate_acceptance}')
        device.notice(
            f'Trial rotate move size: {sim.operations.integrator.a["A"]}')

        # save move size to a file
        if device.communicator.rank == 0:
            name = util.get_job_filename(sim_mode, device, 'move_size', 'json')
            with open(job.fn(name), 'w') as f:
                json.dump(
                    dict(
                        d_A=sim.operations.integrator.d["A"],
                        a_A=sim.operations.integrator.a["A"],
                    ), f)
    else:
        device.notice('Restarting...')
        # read move size from the file
        name = util.get_job_filename(sim_mode, device, 'move_size', 'json')
        with open(job.fn(name), 'r') as f:
            data = json.load(f)

        sim.operations.integrator.d["A"] = data['d_A']
        sim.operations.integrator.a["A"] = data['a_A']
        mcd = sim.operations.integrator.d["A"]
        device.notice(f'Restored translate trial move size: {mcd}')
        mca = sim.operations.integrator.a["A"]
        device.notice(f'Restored rotate trial move size: {mca}')

    # run
    device.notice('Running...')

    util.run_up_to_walltime(sim=sim,
                            end_step=TOTAL_STEPS,
                            steps=RESTART_STEPS,
                            walltime_stop=WALLTIME_STOP_SECONDS)

    hoomd.write.GSD.write(state=sim.state,
                          filename=job.fn(restart_filename),
                          mode='wb')

    if sim.timestep == TOTAL_STEPS:
        pathlib.Path(job.fn(complete_filename)).touch()
        device.notice('Done.')
    else:
        device.notice('Ending run early due to walltime limits at:'
                      f'{device.communicator.walltime}')


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
        initial_state = job.fn('patchy_particle_pressure_initial_state.gsd')
        restart = False

    # box updates
    boxmc = hoomd.hpmc.update.BoxMC(betaP=job.statepoint.pressure,
                                    trigger=hoomd.trigger.Periodic(1))
    boxmc.volume = dict(weight=1.0, mode='ln', delta=1e-6)

    # simulation
    sim = make_mc_simulation(job,
                             device,
                             initial_state,
                             sim_mode,
                             extra_loggables=[
                                 (boxmc, 'volume_moves'),
                             ])

    sim.operations.add(boxmc)

    boxmc_tuner = hoomd.hpmc.tune.BoxMCMoveSize.scale_solver(
        trigger=hoomd.trigger.And([
            hoomd.trigger.Periodic(400),
            hoomd.trigger.Before(RANDOMIZE_STEPS + EQUILIBRATE_STEPS // 2)
        ]),
        boxmc=boxmc,
        moves=['volume'],
        target=0.5)
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
        device.notice(
            f'Translate trial move size: {sim.operations.integrator.d["A"]}')
        rotate_moves = sim.operations.integrator.rotate_moves
        rotate_acceptance = rotate_moves[0] / sum(rotate_moves)
        device.notice(f'Rotate move acceptance: {rotate_acceptance}')
        device.notice(
            f'Rotate trial move size: {sim.operations.integrator.a["A"]}')

        volume_moves = boxmc.volume_moves
        volume_acceptance = volume_moves[0] / sum(volume_moves)
        device.notice(f'Volume move acceptance: {volume_acceptance}')
        device.notice(f'Volume move size: {boxmc.volume["delta"]}')

        # save move sizes to a file
        if device.communicator.rank == 0:
            name = util.get_job_filename(sim_mode, device, 'move_size', 'json')
            with open(job.fn(name), 'w') as f:
                json.dump(
                    dict(d_A=sim.operations.integrator.d["A"],
                         a_A=sim.operations.integrator.a["A"],
                         volume_delta=boxmc.volume['delta']), f)
    else:
        device.notice('Restarting...')
        # read move size from the file
        name = util.get_job_filename(sim_mode, device, 'move_size', 'json')
        with open(job.fn(name), 'r') as f:
            data = json.load(f)

        sim.operations.integrator.d["A"] = data['d_A']
        sim.operations.integrator.a["A"] = data['a_A']
        mcd = sim.operations.integrator.d["A"]
        device.notice(f'Restored translate trial move size: {mcd}')
        mca = sim.operations.integrator.a["A"]
        device.notice(f'Restored rotate trial move size: {mca}')
        boxmc.volume = dict(weight=1.0, mode='ln', delta=data['volume_delta'])
        device.notice(f'Restored volume move size: {boxmc.volume["delta"]}')

    # run
    device.notice('Running...')
    util.run_up_to_walltime(sim=sim,
                            end_step=TOTAL_STEPS,
                            steps=100_000,
                            walltime_stop=WALLTIME_STOP_SECONDS)

    hoomd.write.GSD.write(state=sim.state,
                          filename=job.fn(restart_filename),
                          mode='wb')

    if sim.timestep == TOTAL_STEPS:
        pathlib.Path(job.fn(complete_filename)).touch()
        device.notice('Done.')
    else:
        device.notice('Ending run early due to walltime limits at:'
                      f'{device.communicator.walltime}')


sampling_jobs = []
job_definitions = [
    {
        'mode': 'nvt',
        'device_name': 'cpu',
        'ranks_per_partition': NUM_CPU_RANKS,
        'aggregator': partition_jobs_cpu_mpi_nvt
    },
    {
        'mode': 'npt',
        'device_name': 'cpu',
        'ranks_per_partition': NUM_CPU_RANKS,
        'aggregator': partition_jobs_cpu_mpi_npt
    },
]

#job_definitions = []
if CONFIG["enable_gpu"]:
    job_definitions.extend([
        {
            'mode': 'nvt',
            'device_name': 'gpu',
            'ranks_per_partition': 1,
            'aggregator': partition_jobs_gpu
        },
        {
            'mode': 'npt',
            'device_name': 'gpu',
            'ranks_per_partition': 1,
            'aggregator': partition_jobs_gpu
        },
    ])


def add_sampling_job(mode, device_name, ranks_per_partition, aggregator):
    """Add a sampling job to the workflow."""
    directives = dict(walltime=CONFIG["max_walltime"],
                      executable=CONFIG["executable"],
                      nranks=util.total_ranks_function(ranks_per_partition))

    if device_name == 'gpu':
        directives['ngpu'] = directives['nranks']

    @Project.pre.after(patchy_particle_pressure_create_initial_state)
    @Project.post.isfile(f'{mode}_{device_name}_complete')
    #@Project.post(lambda j: j.isfile(f'{mode}_{device_name}_complete') or j.sp.pressure <= 0)
    @Project.operation(name=f'patchy_particle_pressure_{mode}_{device_name}',
                       directives=directives,
                       aggregator=aggregator)
    def sampling_operation(*jobs):
        """Perform sampling simulation given the definition."""
        import hoomd

        communicator = hoomd.communicator.Communicator(
            ranks_per_partition=ranks_per_partition)
        job = jobs[communicator.partition]

        if communicator.rank == 0:
            print(f'starting patchy_particle_pressure_{mode}_{device_name}:',
                  job)

        if device_name == 'gpu':
            device_cls = hoomd.device.GPU
        elif device_name == 'cpu':
            device_cls = hoomd.device.CPU

        device = device_cls(
            communicator=communicator,
            message_filename=job.fn(f'{mode}_{device_name}.log'))

        globals().get(f'run_{mode}_sim')(
            job, device, complete_filename=f'{mode}_{device_name}_complete')

        if communicator.rank == 0:
            print(f'completed patchy_particle_pressure_{mode}_{device_name} '
                  f'{job} in {communicator.walltime} s')

    sampling_jobs.append(sampling_operation)


for definition in job_definitions:
    add_sampling_job(**definition)


@Project.pre(is_patchy_particle_pressure)
@Project.pre.after(*sampling_jobs)
@Project.post.true('patchy_particle_pressure_analysis_complete')
@Project.operation(directives=dict(walltime=CONFIG['short_walltime'],
                                   executable=CONFIG["executable"]))
def patchy_particle_pressure_analyze(job):
    """Analyze the output of all simulation modes."""
    import gsd.hoomd
    import numpy
    import matplotlib
    import matplotlib.style
    import matplotlib.figure
    matplotlib.style.use('fivethirtyeight')

    print('starting patchy_particle_pressure_analyze:', job)

    sim_modes = []
    for _ensemble in ['nvt', 'npt']:
        for _device in ['cpu', 'gpu']:
            if job.isfile(f'{_ensemble}_{_device}_quantities.gsd'):
                sim_modes.append(f'{_ensemble}_{_device}')

    util._sort_sim_modes(sim_modes)

    timesteps = {}
    pressures = {}
    densities = {}

    for sim_mode in sim_modes:
        log_traj = gsd.hoomd.read_log(job.fn(sim_mode + '_quantities.gsd'))

        timesteps[sim_mode] = log_traj['configuration/step']

        pressures[sim_mode] = log_traj['log/hpmc/compute/SDF/betaP']

        densities[sim_mode] = log_traj[
            'log/custom_actions/ComputeDensity/density']

    # save averages
    for mode in sim_modes:
        job.document[mode] = dict(pressure=float(numpy.mean(pressures[mode])),
                                  density=float(numpy.mean(densities[mode])))

    # Plot results
    fig = matplotlib.figure.Figure(figsize=(10, 10 / 1.618 * 2), layout='tight')
    ax = fig.add_subplot(2, 2, 1)
    util.plot_timeseries(ax=ax,
                         timesteps=timesteps,
                         data=densities,
                         ylabel=r"$\rho$",
                         expected=job.sp.density,
                         max_points=500)
    ax.legend()

    ax_distribution = fig.add_subplot(2, 2, 2, sharey=ax)
    util.plot_distribution(
        ax_distribution,
        {k: v for k, v in densities.items() if not k.startswith('nvt')},
        #densities,
        r'',
        expected=job.sp.density,
        bins=50,
        plot_rotated=True,
    )

    ax = fig.add_subplot(2, 2, 3)
    util.plot_timeseries(ax=ax,
                         timesteps=timesteps,
                         data=pressures,
                         ylabel=r"$\beta P$",
                         expected=job.sp.pressure,
                         max_points=500,)
    ax_distribution = fig.add_subplot(2, 2, 4, sharey=ax)
    util.plot_distribution(
        ax_distribution,
        pressures,
        r'',
        expected=job.sp.pressure,
        bins=50,
        plot_rotated=True,
    )

    fig.suptitle(f"$\\rho={job.statepoint.density}$, "
                 f"$N={job.statepoint.num_particles}$, "
                 f"T={job.statepoint.temperature}, "
                 f"$\\chi={job.statepoint.chi}$, "
                 f"replicate={job.statepoint.replicate_idx}")
    fig.savefig(job.fn('nvt_npt_plots.svg'), bbox_inches='tight', transparent=False)

    job.document['patchy_particle_pressure_analysis_complete'] = True


@Project.pre(lambda *jobs: util.true_all(
    *jobs, key='patchy_particle_pressure_analysis_complete'))
@Project.post(lambda *jobs: util.true_all(
    *jobs, key='patchy_particle_pressure_compare_modes_complete'))
@Project.operation(
    directives=dict(executable=CONFIG["executable"]),
    aggregator=aggregator.groupby(
        key=['pressure', 'density', 'temperature', 'chi', 'num_particles'],
        sort_by='replicate_idx',
        select=is_patchy_particle_pressure))
def patchy_particle_pressure_compare_modes(*jobs):
    """Compares the tested simulation modes."""
    import numpy
    import matplotlib
    import matplotlib.style
    import matplotlib.figure
    matplotlib.style.use('fivethirtyeight')

    print('starting patchy_particle_pressure_compare_modes:', jobs[0])

    sim_modes = []
    for _ensemble in ['nvt', 'npt']:
        for _device in ['cpu', 'gpu']:
            if jobs[0].isfile(f'{_ensemble}_{_device}_quantities.gsd'):
                sim_modes.append(f'{_ensemble}_{_device}')

    util._sort_sim_modes(sim_modes)

    quantity_names = ['density', 'pressure']

    labels = {
        'density': r'$\frac{\rho_\mathrm{sample} - \rho}{\rho} \cdot 1000$',
        'pressure': r'$\frac{P_\mathrm{sample} - P}{P} \cdot 1000$',
    }

    # grab the common statepoint parameters
    set_density = jobs[0].sp.density
    set_pressure = jobs[0].sp.pressure
    set_temperature = jobs[0].sp.temperature
    set_chi = jobs[0].sp.chi
    num_particles = jobs[0].sp.num_particles

    quantity_reference = dict(density=set_density, pressure=set_pressure)

    fig = matplotlib.figure.Figure(figsize=(10, 10 / 1.618 * 2), layout='tight')
    fig.suptitle(
        f"$\\rho={set_density}$, $N={num_particles}$, $T={set_temperature}$, $\\chi={set_chi}$")

    for i, quantity_name in enumerate(quantity_names):
        ax = fig.add_subplot(2, 1, i + 1)

        # organize data from jobs
        quantities = {mode: [] for mode in sim_modes}
        for jb in jobs:
            for mode in sim_modes:
                quantities[mode].append(
                    getattr(getattr(jb.doc, mode), quantity_name))

        if quantity_reference[quantity_name] is not None:
            reference = quantity_reference[quantity_name]
        else:
            avg_value = {
                mode: numpy.mean(quantities[mode]) for mode in sim_modes
            }
            reference = numpy.mean([avg_value[mode] for mode in sim_modes])

        avg_quantity, stderr_quantity = util.plot_vs_expected(
            ax=ax,
            values=quantities,
            ylabel=labels[quantity_name],
            expected=reference,
            relative_scale=1000,
            separate_nvt_npt=True)

    filename = f'patchy_particle_pressure_compare_'
    filename += f'density{round(set_density, 2)}_'
    filename += f'temperature{round(set_temperature, 4)}_'
    filename += f'pressure{round(set_pressure, 3)}_'
    filename += f'chi{round(set_chi, 2)}.svg'
    fig.savefig(os.path.join(jobs[0]._project.path, filename),
                bbox_inches='tight', transparent=False)

    for job in jobs:
        job.document['patchy_particle_pressure_compare_modes_complete'] = True
