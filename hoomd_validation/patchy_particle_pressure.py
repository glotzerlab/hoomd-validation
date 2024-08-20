# Copyright (c) 2022-2024 The Regents of the University of Michigan.
# Part of HOOMD-blue, released under the BSD 3-Clause License.

"""Test for consistency between NVT and NPT simulations of patchy particles."""

import itertools
import json
import os

import hoomd
import matplotlib
import matplotlib.figure
import matplotlib.style
import numpy
import util
from config import CONFIG
from custom_actions import ComputeDensity
from workflow import Action
from workflow_class import ValidationWorkflow

# Run parameters shared between simulations.
# Step counts must be even and a multiple of the log quantity period.
RANDOMIZE_STEPS = 20_000
EQUILIBRATE_STEPS = 200_000
RUN_STEPS = 500_000
RESTART_STEPS = RUN_STEPS // 10
TOTAL_STEPS = RANDOMIZE_STEPS + EQUILIBRATE_STEPS + RUN_STEPS

WRITE_PERIOD = 1_000
LOG_PERIOD = {'trajectory': 50_000, 'quantities': 500}
NUM_CPU_RANKS = min(16, CONFIG['max_cores_sim'])

WALLTIME_STOP_SECONDS = (
    int(os.environ.get('ACTION_WALLTIME_IN_MINUTES', 10)) - 10
) * 60


def job_statepoints():
    """list(dict): A list of statepoints for this subproject."""
    num_particles = 16**3
    replicate_indices = range(CONFIG['replicates'])
    params_list = [
        # kern-frenkel. density and temperature chosen to produce
        # a dense liquid. pressure measured from NVT simulations on CPU.
        (1.0, 0.95, 10.208625410213362, 0.7, 1.5, 1.0),
        # hard sphere + square well, from 10.1063/1.3054361
        # pressure from NVT simulations (on CPU), NOT from values in paper
        (3.0, 0.8, 4.837436833423719, 1.0, 1.5, 1.0),
        # hard sphere + square well + repulsive shoulder.
        # temperatures/densities for dense liquid based on initial tests.
        # pressure measured from NVT simulations on CPU.
        (3.0, 0.6, 2.12496, 1.0, 1.5, -1.0),
    ]  # kT, rho, pressure, chi, lambda_, long_range_interaction_scale_factor
    for temperature, density, pressure, chi, lambda_, lrisf in params_list:
        for idx in replicate_indices:
            yield (
                {
                    'subproject': 'patchy_particle_pressure',
                    'density': density,
                    'pressure': pressure,
                    'chi': chi,
                    'lambda_': lambda_,
                    'num_particles': num_particles,
                    'replicate_idx': idx,
                    'temperature': temperature,
                    'long_range_interaction_scale_factor': lrisf,
                }
            )


_group = {
    'sort_by': ['/density'],
    'include': [{'condition': ['/subproject', '==', __name__]}],
}
_resources = {'walltime': {'per_submission': CONFIG['max_walltime']}}
_resources_cpu = _resources | {'processes': {'per_directory': NUM_CPU_RANKS}}
_group_cpu = _group | {
    'maximum_size': min(
        CONFIG['replicates'], CONFIG['max_cores_submission'] // NUM_CPU_RANKS
    )
}
_group_cpu_postive_pressure = _group_cpu | {
    'include': [{'all': [['/subproject', '==', __name__], ['/pressure', '>', 0]]}]
}

_group_compare = _group | {
    'sort_by': [
        '/pressure',
        '/density',
        '/temperature',
        '/chi',
        '/num_particles',
        '/long_range_interaction_scale_factor',
    ],
    'split_by_sort_key': True,
    'submit_whole': True,
}


def make_potential(
    delta_rad, sq_well_lambda, sigma, kT, long_range_interaction_scale_factor
):
    """Make the Kern-Frenkel pair potential.

    Args:
        delta_rad (float): Half-opening angle of patchy interaction in radians

        sq_well_lambda (float): range of patchy interaction relative to hard
            core diameter

        sigma (float): Diameter of hard core of particles

        kT (float): Temperature; sets the energy scale

        long_range_interaction_scale_factor (float): factor to multiply the
            pair potential by when the interparticle separation is between
            sq_well_lambda/2*sigma and sq_well_lambda*sigma.

    The terminology (e.g., `ehat`) comes from the "Modelling Patchy Particles"
    HOOMD-blue tutorial.
    """
    r = [
        (sigma + sq_well_lambda * sigma) / 2.0,
        sq_well_lambda * sigma,
    ]
    epsilon = [
        -1 / kT,
        -1 / kT * long_range_interaction_scale_factor,
    ]
    step = hoomd.hpmc.pair.Step()
    step.params[('A', 'A')] = dict(epsilon=epsilon, r=r)

    angular_step = hoomd.hpmc.pair.AngularStep(isotropic_potential=step)
    angular_step.mask['A'] = dict(directors=[(1.0, 0, 0)], deltas=[delta_rad])
    return angular_step


def create_initial_state(*jobs):
    """Create initial system configuration."""
    communicator = hoomd.communicator.Communicator(ranks_per_partition=NUM_CPU_RANKS)
    job = jobs[communicator.partition]

    if job.isfile('initial_state.gsd'):
        return

    if communicator.rank == 0:
        print(f'starting {__name__}.create_initial_state:', job)

    num_particles = job.cached_statepoint['num_particles']
    density = job.cached_statepoint['density']
    temperature = job.cached_statepoint['temperature']
    chi = job.cached_statepoint['chi']
    lambda_ = job.cached_statepoint['lambda_']
    long_range_interaction_scale_factor = job.cached_statepoint[
        'long_range_interaction_scale_factor'
    ]

    box_volume = num_particles / density
    L = box_volume ** (1 / 3.0)

    N = int(numpy.ceil(num_particles ** (1.0 / 3.0)))
    x = numpy.linspace(-L / 2, L / 2, N, endpoint=False)

    particle_spacing = 1.0
    if x[1] - x[0] < particle_spacing:
        raise RuntimeError('density too high to initialize on square lattice')

    position = list(itertools.product(x, repeat=3))[:num_particles]

    # create snapshot
    device = hoomd.device.CPU(
        communicator=communicator,
        message_filename=util.get_message_filename(job, 'create_initial_state.log'),
    )
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
    angular_step = make_potential(
        delta,
        lambda_,
        diameter,
        temperature,
        long_range_interaction_scale_factor,
    )
    mc.pair_potentials = [angular_step]

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
        filename=job.fn('initial_state.gsd'),
        mode='wb',
        logger=trajectory_logger,
    )

    if communicator.rank == 0:
        print(f'completed {__name__}.create_initial_state: {job}')


ValidationWorkflow.add_action(
    f'{__name__}.create_initial_state',
    Action(
        method=create_initial_state,
        configuration={
            'products': ['initial_state.gsd'],
            'launchers': ['mpi'],
            'group': _group_cpu,
            'resources': _resources_cpu
            | {'walltime': {'per_submission': CONFIG['short_walltime']}},
        },
    ),
)


def make_mc_simulation(job, device, initial_state, sim_mode, extra_loggables=None):
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
    if extra_loggables is None:
        extra_loggables = []

    # integrator and patchy potential
    temperature = job.cached_statepoint['temperature']
    chi = job.cached_statepoint['chi']
    lambda_ = job.cached_statepoint['lambda_']
    long_range_interaction_scale_factor = job.cached_statepoint[
        'long_range_interaction_scale_factor'
    ]
    diameter = 1.0
    mc = hoomd.hpmc.integrate.Sphere(default_d=0.05, default_a=0.1)
    mc.shape['A'] = dict(diameter=diameter, orientable=True)
    delta = 2 * numpy.arcsin(numpy.sqrt(chi))
    angular_step = make_potential(
        delta,
        lambda_,
        diameter,
        temperature,
        long_range_interaction_scale_factor,
    )
    mc.pair_potentials = [angular_step]

    # compute the density and pressure
    compute_density = ComputeDensity()
    sdf = hoomd.hpmc.compute.SDF(xmax=0.02, dx=1e-4)

    logger = hoomd.logging.Logger(categories=['scalar', 'sequence'])
    logger.add(mc, quantities=['translate_moves'])
    logger.add(sdf, quantities=['betaP'])
    logger.add(compute_density, quantities=['density'])
    logger.add(angular_step, quantities=['energy'])
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


def run_nvt_sim(job, device):
    """Run MC sim in NVT."""
    sim_mode = 'nvt'

    if util.is_simulation_complete(job, device, sim_mode):
        return

    restart_filename = util.get_job_filename(sim_mode, device, 'restart', 'gsd')
    if job.isfile(restart_filename):
        initial_state = job.fn(restart_filename)
        restart = True
    else:
        initial_state = job.fn('initial_state.gsd')
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
        util.mark_simulation_complete(job, device, sim_mode)
        device.notice('Done.')
    else:
        device.notice(f'Ending {job} run early due to walltime limits.')


def run_npt_sim(job, device):
    """Run MC sim in NPT."""
    # device
    sim_mode = 'npt'

    if util.is_simulation_complete(job, device, sim_mode):
        return

    restart_filename = util.get_job_filename(sim_mode, device, 'restart', 'gsd')
    if job.isfile(restart_filename):
        initial_state = job.fn(restart_filename)
        restart = True
    else:
        initial_state = job.fn('initial_state.gsd')
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
        util.mark_simulation_complete(job, device, sim_mode)
        device.notice('Done.')
    else:
        device.notice(f'Ending {job} run early due to walltime limits.')


sampling_jobs = []
job_definitions = [
    {
        'mode': 'nvt',
        'device_name': 'cpu',
        'resources': _resources_cpu,
        'group': _group_cpu,
    },
    {
        'mode': 'npt',
        'device_name': 'cpu',
        'resources': _resources_cpu,
        'group': _group_cpu_postive_pressure,
    },
]


def add_sampling_job(mode, device_name, resources, group):
    """Add a sampling job to the workflow."""
    action_name = f'{__name__}.{mode}_{device_name}'

    def sampling_operation(*jobs):
        """Perform sampling simulation given the definition."""
        communicator = hoomd.communicator.Communicator(
            ranks_per_partition=int(os.environ['ACTION_PROCESSES_PER_DIRECTORY'])
        )
        job = jobs[communicator.partition]

        if communicator.rank == 0:
            print(f'starting {action_name}:', job)

        device = hoomd.device.CPU(
            communicator=communicator,
            message_filename=util.get_message_filename(
                job, f'{mode}_{device_name}.log'
            ),
        )

        globals().get(f'run_{mode}_sim')(job, device)

        if communicator.rank == 0:
            print(f'completed {action_name}: {job}')

    sampling_jobs.append(action_name)

    ValidationWorkflow.add_action(
        action_name,
        Action(
            method=sampling_operation,
            configuration={
                'products': [
                    util.get_job_filename(mode, device_name, 'trajectory', 'gsd'),
                    util.get_job_filename(mode, device_name, 'quantities', 'h5'),
                ],
                'launchers': ['mpi'],
                'group': group,
                'resources': resources,
                'previous_actions': [f'{__name__}.create_initial_state'],
            },
        ),
    )


for definition in job_definitions:
    add_sampling_job(**definition)


def analyze(*jobs):
    """Analyze the output of all simulation modes."""
    matplotlib.style.use('fivethirtyeight')

    for job in jobs:
        print(f'starting {__name__}.analyze:', job)

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
        ax = fig.add_subplot(2, 2, 1)
        util.plot_timeseries(
            ax=ax,
            timesteps=timesteps,
            data=densities,
            ylabel=r'$\rho$',
            expected=job.cached_statepoint['density'],
            max_points=500,
        )
        ax.legend()

        ax_distribution = fig.add_subplot(2, 2, 2, sharey=ax)
        util.plot_distribution(
            ax_distribution,
            {k: v for k, v in densities.items() if not k.startswith('nvt')},
            r'',
            expected=job.cached_statepoint['density'],
            bins=50,
            plot_rotated=True,
        )

        ax = fig.add_subplot(2, 2, 3)
        util.plot_timeseries(
            ax=ax,
            timesteps=timesteps,
            data=pressures,
            ylabel=r'$\beta P$',
            expected=job.cached_statepoint['pressure'],
            max_points=500,
        )
        ax_distribution = fig.add_subplot(2, 2, 4, sharey=ax)
        util.plot_distribution(
            ax_distribution,
            pressures,
            r'',
            expected=job.cached_statepoint['pressure'],
            bins=50,
            plot_rotated=True,
        )

        fig.suptitle(
            f'$\\rho={job.cached_statepoint["density"]}$, '
            f'$N={job.cached_statepoint["num_particles"]}$, '
            f'T={job.cached_statepoint["temperature"]}, '
            f'$\\chi={job.cached_statepoint["chi"]}$, '
            f'replicate={job.cached_statepoint["replicate_idx"]}, '
            '$\\varepsilon_{\\mathrm{rep}}/\\varepsilon_{\\mathrm{att}}$'
            f'$={job.cached_statepoint["long_range_interaction_scale_factor"]}$'
        )
        fig.savefig(job.fn('nvt_npt_plots.svg'), bbox_inches='tight', transparent=False)


ValidationWorkflow.add_action(
    f'{__name__}.analyze',
    Action(
        method=analyze,
        configuration={
            'products': ['nvt_npt_plots.svg'],
            'previous_actions': sampling_jobs,
            'group': _group,
            'resources': {
                'processes': {'per_submission': 1},
                'walltime': {'per_directory': '00:01:00'},
            },
        },
    ),
)


def compare_modes(*jobs):
    """Compares the tested simulation modes."""
    matplotlib.style.use('fivethirtyeight')

    print(f'starting {__name__}.compare_modes:', jobs[0])

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
    set_temperature = jobs[0].sp.temperature
    set_chi = jobs[0].sp.chi
    num_particles = jobs[0].sp.num_particles
    lrisf = jobs[0].sp.long_range_interaction_scale_factor

    quantity_reference = dict(density=set_density, pressure=set_pressure)

    fig = matplotlib.figure.Figure(figsize=(10, 10 / 1.618 * 2), layout='tight')
    fig.suptitle(
        f'$\\rho={set_density}$, '
        f'$N={num_particles}$, '
        f'T={set_temperature}, '
        f'$\\chi={set_chi}$, '
        '$\\varepsilon_{\\mathrm{rep}}/\\varepsilon_{\\mathrm{att}}$'
        f'$={lrisf}$'
    )

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
        ax.axhline(0.0, c='k', ls='--')

        if quantity_name == 'density':
            if 'npt_cpu' in avg_quantity:
                print(
                    f'Average npt_cpu density {num_particles}:',
                    avg_quantity['npt_cpu'],
                    '+/-',
                    stderr_quantity['npt_cpu'],
                )
        if quantity_name == 'pressure':
            if 'nvt_cpu' in avg_quantity:
                print(
                    f'Average nvt_cpu pressure {num_particles}:',
                    avg_quantity['nvt_cpu'],
                    '+/-',
                    stderr_quantity['nvt_cpu'],
                )
            if 'npt_cpu' in avg_quantity:
                print(
                    f'Average npt_cpu pressure {num_particles}:',
                    avg_quantity['npt_cpu'],
                    '+/-',
                    stderr_quantity['npt_cpu'],
                )

    filename = 'patchy_particle_pressure_compare_'
    filename += f'density{round(set_density, 2)}_'
    filename += f'temperature{round(set_temperature, 4)}_'
    filename += f'pressure{round(set_pressure, 3)}_'
    filename += f'chi{round(set_chi, 2)}_'
    filename += f'{num_particles}particles_'
    filename += f'epsilonrepulsive{lrisf}.svg'
    fig.savefig(
        os.path.join(jobs[0]._project.path, filename),
        bbox_inches='tight',
        transparent=False,
    )


ValidationWorkflow.add_action(
    f'{__name__}.compare_modes',
    Action(
        method=compare_modes,
        configuration={
            'previous_actions': [f'{__name__}.analyze'],
            'group': _group_compare,
            'resources': {
                'processes': {'per_submission': 1},
                'walltime': {'per_directory': '00:02:00'},
            },
        },
    ),
)
