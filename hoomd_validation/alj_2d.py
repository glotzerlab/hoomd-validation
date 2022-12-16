# Copyright (c) 2022 The Regents of the University of Michigan.
# Part of HOOMD-blue, released under the BSD 3-Clause License.

"""ALJ 2D energy conservation validation test."""

from config import CONFIG
from project_class import Project
from flow import aggregator
import util
import os

# Run parameters shared between simulations.
# Step counts must be even and a multiple of the log quantity period.
RANDOMIZE_STEPS = 20_000
RUN_STEPS = 300_000_000
TOTAL_STEPS = RANDOMIZE_STEPS + RUN_STEPS

WRITE_PERIOD = 4_000
LOG_PERIOD = {'trajectory': 50_000, 'quantities': 2_000}
ALJ_PARAMS = {'epsilon': 1.0}

# Unit area hexagon
PARTICLE_VERTICES = [[6.20403239e-01, 0.00000000e+00, 0],
                     [3.10201620e-01, 5.37284966e-01, 0],
                     [-3.10201620e-01, 5.37284966e-01, 0],
                     [-6.20403239e-01, 7.59774841e-17, 0],
                     [-3.10201620e-01, -5.37284966e-01, 0],
                     [3.10201620e-01, -5.37284966e-01, 0]]
CIRCUMCIRCLE_RADIUS = 0.6204032392788702
INCIRCLE_RADIUS = 0.5372849659264116
NUM_REPLICATES = 4
NUM_CPU_RANKS = min(8, CONFIG["max_cores_sim"])


def job_statepoints():
    """list(dict): A list of statepoints for this subproject."""
    num_particles = 42**2
    replicate_indices = range(NUM_REPLICATES)
    params_list = [(1.0, 0.4)]
    for kT, density in params_list:
        for idx in replicate_indices:
            yield ({
                "subproject": "alj_2d",
                "kT": kT,
                "density": density,
                "num_particles": num_particles,
                "replicate_idx": idx
            })


def is_alj_2d(job):
    """Test if a given job is part of the alj_2d subproject."""
    return job.statepoint['subproject'] == 'alj_2d'


partition_jobs_cpu = aggregator.groupsof(num=min(
    NUM_REPLICATES, CONFIG["max_cores_submission"] // NUM_CPU_RANKS),
                                         sort_by='density',
                                         select=is_alj_2d)

partition_jobs_gpu = aggregator.groupsof(num=min(NUM_REPLICATES,
                                                 CONFIG["max_gpus_submission"]),
                                         sort_by='density',
                                         select=is_alj_2d)


@Project.post.isfile('alj_2d_initial_state.gsd')
@Project.operation(directives=dict(
    executable=CONFIG["executable"],
    nranks=util.total_ranks_function(NUM_CPU_RANKS),
    walltime=1),
                   aggregator=partition_jobs_cpu)
def alj_2d_create_initial_state(*jobs):
    """Create initial system configuration."""
    import hoomd
    import numpy
    import itertools

    communicator = hoomd.communicator.Communicator(
        ranks_per_partition=NUM_CPU_RANKS)
    job = jobs[communicator.partition]

    if communicator.rank == 0:
        print('starting alj2_create_initial_state:', job)

    init_diameter = CIRCUMCIRCLE_RADIUS * 2 * 1.15

    device = hoomd.device.CPU(communicator=communicator,
                              msg_file=job.fn('create_initial_state.log'))

    num_particles = job.statepoint['num_particles']
    density = job.statepoint['density']

    box_volume = num_particles / density
    L = box_volume**(1 / 2.)

    N = int(numpy.ceil(num_particles**(1. / 2.)))
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

    sim = hoomd.Simulation(device=device, seed=job.statepoint.replicate_idx)
    sim.create_state_from_snapshot(snap)
    sim.operations.integrator = mc

    device.notice('Randomizing initial state...')
    sim.run(RANDOMIZE_STEPS)
    device.notice(f'Done. Move counts: {mc.translate_moves}')

    hoomd.write.GSD.write(state=sim.state,
                          filename=job.fn("alj_2d_initial_state.gsd"),
                          mode='wb')


def make_md_simulation(job,
                       device,
                       initial_state,
                       method,
                       sim_mode,
                       period_multiplier=1):
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
    import hoomd
    from hoomd import md

    incircle_d = INCIRCLE_RADIUS * 2
    circumcircle_d = CIRCUMCIRCLE_RADIUS * 2
    r_cut = max(2**(1 / 6) * incircle_d,
                circumcircle_d + 2**(1 / 6) * 0.15 * incircle_d)

    # pair force
    nlist = md.nlist.Cell(buffer=0.4)
    alj = md.pair.aniso.ALJ(default_r_cut=r_cut, nlist=nlist)
    alj.shape['A'] = {
        "vertices": PARTICLE_VERTICES,
        "faces": [],
        "rounding_radii": 0
    }
    alj.params[("A", "A")] = {
        "epsilon": ALJ_PARAMS['epsilon'],
        "sigma_i": incircle_d,
        "sigma_j": incircle_d,
        "alpha": 0
    }

    # integrator
    integrator = md.Integrator(dt=0.0001,
                               methods=[method],
                               forces=[alj],
                               integrate_rotational_dof=True)

    # compute thermo
    thermo = md.compute.ThermodynamicQuantities(hoomd.filter.All())

    # add gsd log quantities
    logger = hoomd.logging.Logger(categories=['scalar', 'sequence'])
    logger.add(thermo,
               quantities=[
                   'pressure',
                   'potential_energy',
                   'kinetic_temperature',
                   'kinetic_energy',
                   'translational_kinetic_energy',
                   'rotational_kinetic_energy',
               ])
    logger.add(integrator, quantities=['linear_momentum'])

    # simulation
    sim = util.make_simulation(job=job, device=device, initial_state=initial_state, integrator=integrator, sim_mode=sim_mode,
                               logger=logger, table_write_period=WRITE_PERIOD,
                               trajectory_write_period=LOG_PERIOD['trajectory'] * period_multiplier,
                               log_write_period=LOG_PERIOD['quantities'] * period_multiplier,
                               log_start_step=RANDOMIZE_STEPS)
    sim.operations.add(thermo)

    # thermalize momenta
    sim.state.thermalize_particle_momenta(hoomd.filter.All(), job.sp.kT)

    return sim


def run_nve_md_sim(job, device):
    """Run the MD simulation in NVE."""
    import hoomd

    initial_state = job.fn('alj_2d_initial_state.gsd')
    nvt = hoomd.md.methods.NVE(hoomd.filter.All())
    sim_mode = 'nve_md'

    sim = make_md_simulation(job,
                             device,
                             initial_state,
                             nvt,
                             sim_mode,
                             period_multiplier=300)

    # Run for a long time to look for energy and momentum drift
    device.notice('Running...')

    util.run_up_to_walltime(sim=sim, end_step=TOTAL_STEPS, steps=500_000, walltime_stop= CONFIG["max_walltime"] * 3600 - 10 * 60)

    if sim.timestep == TOTAL_STEPS:
        device.notice('Done.')
    else:
        device.notice('Ending run early due to walltime limits at:',
                      device.communicator.walltime)

    device.notice('Done.')


@Project.pre.after(alj_2d_create_initial_state)
@Project.post(util.gsd_step_greater_equal_function('nve_cpu_quantities.gsd', TOTAL_STEPS))
@Project.operation(directives=dict(
    walltime=CONFIG["max_walltime"],
    executable=CONFIG["executable"],
    nranks=util.total_ranks_function(NUM_CPU_RANKS)),
                   aggregator=partition_jobs_cpu)
def alj_2d_nve_md_cpu(*jobs):
    """Run NVE MD on the CPU."""
    import hoomd

    communicator = hoomd.communicator.Communicator(
        ranks_per_partition=NUM_CPU_RANKS)
    job = jobs[communicator.partition]

    if communicator.rank == 0:
        print('starting alj2_2d_nve_md_cpu:', job)

    device = hoomd.device.CPU(communicator=communicator,
                              msg_file=job.fn('run_nve_md_cpu.log'))
    run_nve_md_sim(job, device)


@Project.pre.after(alj_2d_create_initial_state)
@Project.post(util.gsd_step_greater_equal_function('nve_gpu_quantities.gsd', TOTAL_STEPS))
@Project.operation(directives=dict(walltime=CONFIG["max_walltime"],
                                   executable=CONFIG["executable"],
                                   nranks=util.total_ranks_function(1),
                                   ngpu=util.total_ranks_function(1)),
                   aggregator=partition_jobs_gpu)
def alj_2d_nve_md_gpu(*jobs):
    """Run NVE MD on the GPU."""
    import hoomd

    communicator = hoomd.communicator.Communicator(ranks_per_partition=1)
    job = jobs[communicator.partition]

    if communicator.rank == 0:
        print('starting alj2_2d_nve_md_gpu:', job)

    device = hoomd.device.GPU(communicator=communicator,
                              msg_file=job.fn('run_nve_md_gpu.log'))
    run_nve_md_sim(job, device)


analysis_aggregator = aggregator.groupby(key=['kT', 'density', 'num_particles'],
                                         sort_by='replicate_idx',
                                         select=is_alj_2d)


@Project.pre(util.gsd_step_greater_equal_function('nve_cpu_quantities.gsd', TOTAL_STEPS))
@Project.pre(util.gsd_step_greater_equal_function('nve_gpu_quantities.gsd', TOTAL_STEPS))
@Project.post(lambda *jobs: util.true_all(
    *jobs, key='alj_2d_conservation_analysis_complete'))
@Project.operation(directives=dict(walltime=1, executable=CONFIG["executable"]),
                   aggregator=analysis_aggregator)
def alj_2d_conservation_analyze(*jobs):
    """Analyze the output of NVE simulations and inspect conservation."""
    import gsd.hoomd
    import numpy
    import math
    import matplotlib
    import matplotlib.style
    import matplotlib.figure
    matplotlib.style.use('ggplot')
    from util import read_gsd_log_trajectory, get_log_quantity

    print('starting alj_2d_conservation_analyze:', jobs[0])

    sim_modes = ['nve_md_cpu', 'nve_md_gpu']

    energies = []
    linear_momenta = []

    for job in jobs:
        job_energies = {}
        job_linear_momentum = {}

        for sim_mode in sim_modes:
            with gsd.hoomd.open(job.fn(sim_mode
                                       + '_quantities.gsd')) as gsd_traj:
                # read GSD file once
                traj = read_gsd_log_trajectory(gsd_traj)

            job_energies[sim_mode] = (
                get_log_quantity(
                    traj, 'md/compute/ThermodynamicQuantities/potential_energy')
                + get_log_quantity(
                    traj, 'md/compute/ThermodynamicQuantities/kinetic_energy'))
            job_energies[sim_mode] = (
                job_energies[sim_mode]
                - job_energies[sim_mode][0]) / job.statepoint["num_particles"]

            momentum_vector = get_log_quantity(traj,
                                               'md/Integrator/linear_momentum')
            job_linear_momentum[sim_mode] = [
                math.sqrt(v[0]**2 + v[1]**2 + v[2]**2) for v in momentum_vector
            ]

        energies.append(job_energies)

        linear_momenta.append(job_linear_momentum)

    # Plot results
    def plot(*, ax, data, quantity_name, legend=False):
        for i, job in enumerate(jobs):
            for mode in sim_modes:
                ax.plot(numpy.asarray(data[i][mode]),
                        label=f'{mode}_{job.statepoint.replicate_idx}')
        ax.set_xlabel("frame")
        ax.set_ylabel(quantity_name)

        if legend:
            ax.legend()

    fig = matplotlib.figure.Figure(figsize=(10, 10 / 1.68 * 2), layout='tight')
    ax = fig.add_subplot(2, 1, 1)
    plot(ax=ax, data=energies, quantity_name=r"$E / N$", legend=True)

    ax = fig.add_subplot(2, 1, 2)
    plot(ax=ax, data=linear_momenta, quantity_name=r"$p$")

    fig.suptitle("ALJ 2D conservation tests: "
                 f"$kT={job.statepoint.kT}$, $\\rho={job.statepoint.density}$, "
                 f"$N={job.statepoint.num_particles}$")
    filename = f'alj_2d_conservation_kT{job.statepoint.kT}_' \
               f'density{round(job.statepoint.density, 2)}_' \
               f'N{job.statepoint.num_particles}.svg'

    fig.savefig(os.path.join(jobs[0]._project.path, filename),
                bbox_inches='tight')

    for job in jobs:
        job.document['alj_2d_conservation_analysis_complete'] = True
