# Copyright (c) 2022-2024 The Regents of the University of Michigan.
# Part of HOOMD-blue, released under the BSD 3-Clause License.

"""Hard disk equation of state validation test."""

import itertools
import json
import os

try:
    import hoomd
except ModuleNotFoundError as e:
    print(f"Warning: {e}")

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
EQUILIBRATE_STEPS = 100_000
RUN_STEPS = 500_000
RESTART_STEPS = RUN_STEPS // 10
TOTAL_STEPS = RANDOMIZE_STEPS + EQUILIBRATE_STEPS + RUN_STEPS

WRITE_PERIOD = 1_000
LOG_PERIOD = {'trajectory': 50_000, 'quantities': 100}
NUM_CPU_RANKS = min(64, CONFIG['max_cores_sim'])

WALLTIME_STOP_SECONDS = (
    int(os.environ.get('ACTION_WALLTIME_IN_MINUTES', 10)) - 10
) * 60


def job_statepoints():
    """list(dict): A list of statepoints for this subproject."""
    num_particles = 128**2
    replicate_indices = range(CONFIG['replicates'])
    # reference statepoint from: http://dx.doi.org/10.1016/j.jcp.2013.07.023
    # Assume the same pressure for a 128**2 system as this is in the fluid
    params_list = [(0.8887212022251435, 9.17079)]
    for density, pressure in params_list:
        for idx in replicate_indices:
            yield (
                {
                    'subproject': 'hard_disk',
                    'density': density,
                    'pressure': pressure,
                    'num_particles': num_particles,
                    'replicate_idx': idx,
                }
            )


_group = {
    'sort_by': ['/density'],
    'include': [{'condition': ['/subproject', '==', __name__]}],
}
_resources = {'walltime': {'per_submission': CONFIG['max_walltime']}}
_resources_serial = _resources | {'processes': {'per_directory': 1}}
_group_serial = _group | {
    'maximum_size': min(CONFIG['replicates'], CONFIG['max_cores_submission'])
}
_resources_cpu = _resources | {'processes': {'per_directory': NUM_CPU_RANKS}}
_group_cpu = _group | {
    'maximum_size': min(
        CONFIG['replicates'], CONFIG['max_cores_submission'] // NUM_CPU_RANKS
    )
}
_resources_gpu = _resources | {'processes': {'per_directory': 1}, 'gpus_per_process': 1}
_group_gpu = _group | {'maximum_size': CONFIG['max_gpus_submission']}
_group_compare = _group | {
    'sort_by': ['/kT', '/density', '/num_particles'],
    'split_by_sort_key': True,
    'submit_whole': True,
}

_group_compare = _group | {
    'sort_by': ['/density', '/num_particles'],
    'split_by_sort_key': True,
    'submit_whole': True,
}


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
    mc = hoomd.hpmc.integrate.Sphere(default_d=0.01)
    mc.shape['A'] = dict(diameter=1.0)

    sim = hoomd.Simulation(device=device, seed=util.make_seed(job))
    sim.create_state_from_snapshot(snap)
    sim.operations.integrator = mc

    device.notice('Randomizing initial state...')
    sim.run(RANDOMIZE_STEPS)
    device.notice(f'Move counts: {mc.translate_moves}')
    device.notice('Done.')

    hoomd.write.GSD.write(
        state=sim.state, filename=job.fn('initial_state.gsd'), mode='wb'
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
    """Make a hard sphere MC Simulation.

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

    # integrator
    mc = hoomd.hpmc.integrate.Sphere(nselect=4)
    mc.shape['A'] = dict(diameter=1.0)

    # compute the density and pressure
    compute_density = ComputeDensity()
    sdf = hoomd.hpmc.compute.SDF(xmax=0.02, dx=1e-4)

    logger = hoomd.logging.Logger(categories=['scalar', 'sequence'])
    logger.add(mc, quantities=['translate_moves'])
    logger.add(sdf, quantities=['betaP'])
    logger.add(compute_density, quantities=['density'])
    for loggable, quantity in extra_loggables:
        logger.add(loggable, quantities=[quantity])

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
        moves=['d'],
        target=0.2,
        max_translation_move=0.5,
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
        device.notice(f'Translate move acceptance: {translate_acceptance}')
        device.notice(f'Trial move size: {sim.operations.integrator.d["A"]}')

        # save move size to a file
        if device.communicator.rank == 0:
            name = util.get_job_filename(sim_mode, device, 'move_size', 'json')
            with open(job.fn(name), 'w') as f:
                json.dump(dict(d_A=sim.operations.integrator.d['A']), f)
    else:
        device.notice('Restarting...')
        # read move size from the file
        name = util.get_job_filename(sim_mode, device, 'move_size', 'json')
        with open(job.fn(name)) as f:
            data = json.load(f)

        sim.operations.integrator.d['A'] = data['d_A']
        device.notice(f'Restored trial move size: {sim.operations.integrator.d["A"]}')

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
        device.notice(
            'Ending run early due to walltime limits at:'
            f'{device.communicator.walltime}'
        )


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
        device.notice(f'Trial move size: {sim.operations.integrator.d["A"]}')

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
        device.notice(f'Restored trial move size: {sim.operations.integrator.d["A"]}')
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
        device.notice(
            'Ending run early due to walltime limits at:'
            f'{device.communicator.walltime}'
        )


def run_nec_sim(job, device):
    """Run MC sim in NVT with NEC."""
    sim_mode = 'nec'

    if util.is_simulation_complete(job, device, sim_mode):
        return

    restart_filename = util.get_job_filename(sim_mode, device, 'restart', 'gsd')
    if job.isfile(restart_filename):
        initial_state = job.fn(restart_filename)
        restart = True
    else:
        initial_state = job.fn('initial_state.gsd')
        restart = False

    mc = hoomd.hpmc.nec.integrate.Sphere(
        default_d=0.05, update_fraction=0.005, nselect=1
    )
    mc.shape['A'] = dict(diameter=1)
    mc.chain_time = 0.05

    # compute the density
    compute_density = ComputeDensity()

    logger = hoomd.logging.Logger(categories=['scalar', 'sequence'])
    logger.add(
        mc, quantities=['translate_moves', 'particles_per_chain', 'virial_pressure']
    )
    logger.add(compute_density, quantities=['density'])

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
    )

    compute_density.attach(sim)

    trigger_tune = hoomd.trigger.And(
        [
            hoomd.trigger.Periodic(5),
            hoomd.trigger.Before(RANDOMIZE_STEPS + EQUILIBRATE_STEPS // 2),
        ]
    )

    tune_nec_d = hoomd.hpmc.tune.MoveSize.scale_solver(
        trigger=trigger_tune,
        moves=['d'],
        target=0.10,
        tol=0.001,
        max_translation_move=0.25,
    )
    sim.operations.tuners.append(tune_nec_d)

    tune_nec_ct = hoomd.hpmc.nec.tune.ChainTime.scale_solver(
        trigger_tune, target=20.0, tol=1.0, gamma=20.0
    )
    sim.operations.tuners.append(tune_nec_ct)

    # equilibrate
    if not restart:
        sim.state.thermalize_particle_momenta(hoomd.filter.All(), kT=1)

        device.notice('Equilibrating...')
        sim.run(EQUILIBRATE_STEPS // 2)
        sim.run(EQUILIBRATE_STEPS // 2)
        device.notice('Done.')

        # Print acceptance ratio as measured during the 2nd half of the
        # equilibration.
        translate_moves = sim.operations.integrator.translate_moves
        translate_acceptance = translate_moves[0] / sum(translate_moves)
        device.notice(f'Collision search acceptance: {translate_acceptance}')
        device.notice(f'Collision search size: {sim.operations.integrator.d["A"]}')
        device.notice(f'Particles per chain: {mc.particles_per_chain}')
        device.notice(f'Chain time: {mc.chain_time}')

        # save move sizes to a file
        if device.communicator.rank == 0:
            name = util.get_job_filename(sim_mode, device, 'move_size', 'json')
            with open(job.fn(name), 'w') as f:
                json.dump(
                    dict(
                        d_A=sim.operations.integrator.d['A'], chain_time=mc.chain_time
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
        device.notice(
            'Restored collision search size: ' f'{sim.operations.integrator.d["A"]}'
        )
        mc.chain_time = data['chain_time']
        device.notice(f'Restored chain time: {mc.chain_time}')

    # run
    device.notice('Running...')

    # limit the run length to the given walltime
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
        device.notice(
            'Ending run early due to walltime limits at:'
            f'{device.communicator.walltime}'
        )


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
        'group': _group_cpu,
    },
    {
        'mode': 'nec',
        'device_name': 'cpu',
        'resources': _resources_serial,
        'group': _group_serial,
    },
]

if CONFIG['enable_gpu']:
    job_definitions.extend(
        [
            {
                'mode': 'nvt',
                'device_name': 'gpu',
                'resources': _resources_gpu,
                'group': _group_gpu,
            },
        ]
    )


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

        if device_name == 'gpu':
            device_cls = hoomd.device.GPU
        elif device_name == 'cpu':
            device_cls = hoomd.device.CPU

        device = device_cls(
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

        sim_modes = [
            'nvt_cpu',
            'nec_cpu',
            'npt_cpu',
        ]

        if os.path.exists(job.fn('nvt_gpu_quantities.h5')):
            sim_modes.extend(['nvt_gpu'])

        util._sort_sim_modes(sim_modes)

        timesteps = {}
        pressures = {}
        densities = {}

        for sim_mode in sim_modes:
            log_traj = util.read_log(job.fn(sim_mode + '_quantities.h5'))

            timesteps[sim_mode] = log_traj['hoomd-data/Simulation/timestep']

            if 'nec' in sim_mode:
                pressures[sim_mode] = log_traj[
                    'hoomd-data/hpmc/nec/integrate/Sphere/virial_pressure'
                ]
            else:
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
            f'$\\rho={job.cached_statepoint["density"]}$, '
            f'$N={job.cached_statepoint["num_particles"]}$, '
            f'replicate={job.cached_statepoint["replicate_idx"]}'
        )
        fig.savefig(job.fn('nvt_npt_plots.svg'), bbox_inches='tight')


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

    sim_modes = [
        'nvt_cpu',
        'nec_cpu',
        'npt_cpu',
    ]

    if os.path.exists(jobs[0].fn('nvt_gpu_quantities.h5')):
        sim_modes.extend(['nvt_gpu'])

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

        util.plot_vs_expected(
            ax=ax,
            values=quantities,
            ylabel=labels[quantity_name],
            expected=reference,
            relative_scale=1000,
            separate_nvt_npt=True,
        )

    filename = f'hard_disk_compare_density{round(set_density, 2)}.svg'
    fig.savefig(os.path.join(jobs[0]._project.path, filename), bbox_inches='tight')


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
