# Copyright (c) 2022-2024 The Regents of the University of Michigan.
# Part of HOOMD-blue, released under the BSD 3-Clause License.

"""ALJ 2D energy conservation validation test."""

import itertools
import math
import os

try:
    import hoomd
except ModuleNotFoundError as e:
    print(f'Warning: {e}')

import matplotlib
import matplotlib.figure
import matplotlib.style
import numpy
import util
from config import CONFIG
from workflow import Action
from workflow_class import ValidationWorkflow

# Run parameters shared between simulations.
# Step counts must be even and a multiple of the log quantity period.
RANDOMIZE_STEPS = 20_000
RUN_STEPS = 50_000_000
RESTART_STEPS = RUN_STEPS // 100
TOTAL_STEPS = RANDOMIZE_STEPS + RUN_STEPS

WRITE_PERIOD = 4_000
LOG_PERIOD = {'trajectory': 50_000, 'quantities': 1_000}
ALJ_PARAMS = {'epsilon': 1.0}

# Unit area hexagon
PARTICLE_VERTICES = [
    [6.20403239e-01, 0.00000000e00, 0],
    [3.10201620e-01, 5.37284966e-01, 0],
    [-3.10201620e-01, 5.37284966e-01, 0],
    [-6.20403239e-01, 7.59774841e-17, 0],
    [-3.10201620e-01, -5.37284966e-01, 0],
    [3.10201620e-01, -5.37284966e-01, 0],
]
CIRCUMCIRCLE_RADIUS = 0.6204032392788702
INCIRCLE_RADIUS = 0.5372849659264116
NUM_REPLICATES = min(4, CONFIG['replicates'])
NUM_CPU_RANKS = min(8, CONFIG['max_cores_sim'])

WALLTIME_STOP_SECONDS = (
    int(os.environ.get('ACTION_WALLTIME_IN_MINUTES', 10)) - 10
) * 60


def job_statepoints():
    """list(dict): A list of statepoints for this subproject."""
    num_particles = 42**2
    replicate_indices = range(NUM_REPLICATES)
    params_list = [(1.0, 0.4)]
    for kT, density in params_list:
        for idx in replicate_indices:
            yield (
                {
                    'subproject': 'alj_2d',
                    'kT': kT,
                    'density': density,
                    'num_particles': num_particles,
                    'replicate_idx': idx,
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
_resources_gpu = _resources | {'processes': {'per_directory': 1}, 'gpus_per_process': 1}
_group_gpu = _group | {'maximum_size': CONFIG['max_gpus_submission']}

_group_compare = _group | {
    'sort_by': ['/kT', '/density', '/num_particles'],
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

    init_diameter = CIRCUMCIRCLE_RADIUS * 2 * 1.15

    device = hoomd.device.CPU(
        communicator=communicator,
        message_filename=util.get_message_filename(job, 'create_initial_state.log'),
    )

    num_particles = job.cached_statepoint['num_particles']
    density = job.cached_statepoint['density']

    box_volume = num_particles / density
    L = box_volume ** (1 / 2.0)

    N = int(numpy.ceil(num_particles ** (1.0 / 2.0)))
    x = numpy.linspace(-L / 2, L / 2, N, endpoint=False)

    if x[1] - x[0] < init_diameter:
        raise RuntimeError('density too high to initialize on square lattice')

    position_2d = list(itertools.product(x, repeat=2))[:num_particles]

    # create snapshot
    snap = hoomd.Snapshot(device.communicator)

    if device.communicator.rank == 0:
        snap.particles.N = num_particles
        snap.particles.types = ['A']
        snap.configuration.box = [L, L, 0, 0, 0, 0]
        snap.particles.position[:, 0:2] = position_2d
        snap.particles.typeid[:] = [0] * num_particles
        snap.particles.moment_inertia[:] = [0, 0, 1]

    # Use hard sphere Monte-Carlo to randomize the initial configuration
    mc = hoomd.hpmc.integrate.Sphere()
    mc.shape['A'] = dict(diameter=init_diameter)

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


def make_md_simulation(
    job, device, initial_state, method, sim_mode, period_multiplier=1
):
    """Make an MD simulation.

    Args:
        job (`signac.job.Job`): Signac job object.

        device (`hoomd.device.Device`): hoomd device object.

        initial_state (str): Path to the gsd file to be used as an initial state
            for the simulation.

        method (`hoomd.md.methods.Method`): hoomd integration method.

        sim_mode (str): String identifying the simulation mode.

        ThermodynamicQuantities is added by default, any more quantities should
            be in this list.

        period_multiplier (int): Factor to multiply the GSD file periods by.
    """
    incircle_d = INCIRCLE_RADIUS * 2
    circumcircle_d = CIRCUMCIRCLE_RADIUS * 2
    r_cut = max(
        2 ** (1 / 6) * incircle_d, circumcircle_d + 2 ** (1 / 6) * 0.15 * incircle_d
    )

    # pair force
    nlist = hoomd.md.nlist.Cell(buffer=0.4)
    alj = hoomd.md.pair.aniso.ALJ(default_r_cut=r_cut, nlist=nlist)
    alj.shape['A'] = {'vertices': PARTICLE_VERTICES, 'faces': [], 'rounding_radii': 0}
    alj.params[('A', 'A')] = {
        'epsilon': ALJ_PARAMS['epsilon'],
        'sigma_i': incircle_d,
        'sigma_j': incircle_d,
        'alpha': 0,
    }

    # integrator
    integrator = hoomd.md.Integrator(
        dt=0.0001, methods=[method], forces=[alj], integrate_rotational_dof=True
    )

    # compute thermo
    thermo = hoomd.md.compute.ThermodynamicQuantities(hoomd.filter.All())

    # add gsd log quantities
    logger = hoomd.logging.Logger(categories=['scalar', 'sequence'])
    logger.add(
        thermo,
        quantities=[
            'pressure',
            'potential_energy',
            'kinetic_temperature',
            'kinetic_energy',
            'translational_kinetic_energy',
            'rotational_kinetic_energy',
        ],
    )
    logger.add(integrator, quantities=['linear_momentum'])

    # simulation
    sim = util.make_simulation(
        job=job,
        device=device,
        initial_state=initial_state,
        integrator=integrator,
        sim_mode=sim_mode,
        logger=logger,
        table_write_period=WRITE_PERIOD,
        trajectory_write_period=LOG_PERIOD['trajectory'] * period_multiplier,
        log_write_period=LOG_PERIOD['quantities'] * period_multiplier,
        log_start_step=RANDOMIZE_STEPS,
    )
    sim.operations.add(thermo)

    # thermalize momenta
    sim.state.thermalize_particle_momenta(
        hoomd.filter.All(), job.cached_statepoint['kT']
    )

    return sim


def run_nve_md_sim(job, device):
    """Run the MD simulation in NVE."""
    sim_mode = 'nve_md'

    if util.is_simulation_complete(job, device, sim_mode):
        return

    restart_filename = util.get_job_filename(sim_mode, device, 'restart', 'gsd')
    if job.isfile(restart_filename):
        initial_state = job.fn(restart_filename)
    else:
        initial_state = job.fn('initial_state.gsd')

    nve = hoomd.md.methods.ConstantVolume(hoomd.filter.All())

    sim = make_md_simulation(
        job, device, initial_state, nve, sim_mode, period_multiplier=50
    )

    # Run for a long time to look for energy and momentum drift
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


nve_md_sampling_jobs = []
nve_md_job_definitions = [
    {
        'device_name': 'cpu',
    },
]

if CONFIG['enable_gpu']:
    nve_md_job_definitions.extend(
        [
            {
                'device_name': 'gpu',
            },
        ]
    )


def add_nve_md_job(device_name):
    """Add a MD NVE conservation job to the workflow."""
    sim_mode = 'nve_md'
    action_name = f'{__name__}.{sim_mode}_{device_name}'

    def nve_action(*jobs):
        """Run NVE MD."""
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
                job, f'{sim_mode}_{device_name}.log'
            ),
        )
        run_nve_md_sim(job, device)

        if communicator.rank == 0:
            print(f'completed {action_name}: {job}')

    nve_md_sampling_jobs.append(action_name)

    ValidationWorkflow.add_action(
        action_name,
        Action(
            method=nve_action,
            configuration={
                'products': [
                    util.get_job_filename(sim_mode, device_name, 'trajectory', 'gsd'),
                    util.get_job_filename(sim_mode, device_name, 'quantities', 'h5'),
                ],
                'launchers': ['mpi'],
                'group': globals().get(f'_group_{device_name}'),
                'resources': globals().get(f'_resources_{device_name}'),
                'previous_actions': [f'{__name__}.create_initial_state'],
            },
        ),
    )


for definition in nve_md_job_definitions:
    add_nve_md_job(**definition)


def conservation_analyze(*jobs):
    """Analyze the output of NVE simulations and inspect conservation."""
    matplotlib.style.use('fivethirtyeight')

    print(f'starting {__name__}.conservation_analyze:', jobs[0])

    sim_modes = ['nve_md_cpu']
    if os.path.exists(jobs[0].fn('nve_md_gpu_quantities.h5')):
        sim_modes.extend(['nve_md_gpu'])

    timesteps = []
    energies = []
    linear_momenta = []

    for job in jobs:
        job_timesteps = {}
        job_energies = {}
        job_linear_momentum = {}

        for sim_mode in sim_modes:
            log_traj = util.read_log(job.fn(sim_mode + '_quantities.h5'))

            job_timesteps[sim_mode] = log_traj['hoomd-data/Simulation/timestep']

            job_energies[sim_mode] = (
                log_traj[
                    'hoomd-data/md/compute/ThermodynamicQuantities/potential_energy'
                ]
                + log_traj[
                    'hoomd-data/md/compute/ThermodynamicQuantities/kinetic_energy'
                ]
            )
            job_energies[sim_mode] = (
                job_energies[sim_mode] - job_energies[sim_mode][0]
            ) / job.cached_statepoint['num_particles']

            momentum_vector = log_traj['hoomd-data/md/Integrator/linear_momentum']
            job_linear_momentum[sim_mode] = [
                math.sqrt(v[0] ** 2 + v[1] ** 2 + v[2] ** 2)
                / job.cached_statepoint['num_particles']
                for v in momentum_vector
            ]

        timesteps.append(job_timesteps)
        energies.append(job_energies)
        linear_momenta.append(job_linear_momentum)

    # Plot results
    def plot(*, ax, data, quantity_name, legend=False):
        for i, job in enumerate(jobs):
            for mode in sim_modes:
                ax.plot(
                    timesteps[i][mode],
                    data[i][mode],
                    label=f'{mode}_{job.cached_statepoint["replicate_idx"]}',
                )
        ax.set_xlabel('time step')
        ax.set_ylabel(quantity_name)

        if legend:
            ax.legend()

    fig = matplotlib.figure.Figure(figsize=(10, 10 / 1.68 * 2), layout='tight')
    ax = fig.add_subplot(2, 1, 1)
    plot(ax=ax, data=energies, quantity_name=r'$E / N$', legend=True)

    ax = fig.add_subplot(2, 1, 2)
    plot(ax=ax, data=linear_momenta, quantity_name=r'$\left| \vec{p} \right| / N$')

    fig.suptitle(
        'ALJ 2D conservation tests: '
        f'$kT={job.cached_statepoint["kT"]}$, '
        f'$\\rho={job.cached_statepoint["density"]}$, '
        f'$N={job.cached_statepoint["num_particles"]}$'
    )
    filename = (
        f'alj_2d_conservation_kT{job.cached_statepoint["kT"]}_'
        f'density{round(job.cached_statepoint["density"], 2)}_'
        f'N{job.cached_statepoint["num_particles"]}.svg'
    )

    fig.savefig(os.path.join(jobs[0]._project.path, filename), bbox_inches='tight')


ValidationWorkflow.add_action(
    f'{__name__}.conservation_analyze',
    Action(
        method=conservation_analyze,
        configuration={
            'previous_actions': nve_md_sampling_jobs,
            'group': _group_compare,
            'resources': {
                'processes': {'per_submission': 1},
                'walltime': {'per_directory': '00:02:00'},
            },
        },
    ),
)
