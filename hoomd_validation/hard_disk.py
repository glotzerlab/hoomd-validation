# Copyright (c) 2022 The Regents of the University of Michigan.
# Part of HOOMD-blue, released under the BSD 3-Clause License.

"""Hard disk equation of state validation test."""

from config import CONFIG
from project_class import Project
from flow import aggregator
import util
import os

# Run parameters shared between simulations
RANDOMIZE_STEPS = 5e4
RUN_STEPS = 2e6
WRITE_PERIOD = 1000
LOG_PERIOD = {'trajectory': 50000, 'quantities': 125}
FRAMES_ANALYZE = int(RUN_STEPS / LOG_PERIOD['quantities'] * 1 / 2)


def job_statepoints():
    """list(dict): A list of statepoints for this subproject."""
    num_particles = 256**2
    replicate_indices = range(8)
    # reference statepoint from: http://dx.doi.org/10.1016/j.jcp.2013.07.023
    params_list = [(0.8887212022251435, 9.17079)]
    for density, pressure in params_list:
        for idx in replicate_indices:
            yield ({
                "subproject": "hard_disk",
                "density": density,
                "pressure": pressure,
                "num_particles": num_particles,
                "replicate_idx": idx
            })


def is_hard_disk(job):
    """Test if a given job is part of the hard_disk subproject."""
    return job.statepoint['subproject'] == 'hard_disk'


@Project.operation(directives=dict(executable=CONFIG["executable"],
                                   nranks=1,
                                   walltime=1))
@Project.pre(is_hard_disk)
@Project.post.isfile('hard_disk_initial_state.gsd')
def hard_disk_create_initial_state(job):
    """Create initial system configuration."""
    import gsd.hoomd
    import numpy
    import itertools

    num_particles = job.statepoint['num_particles']
    density = job.statepoint['density']

    box_volume = num_particles / density
    L = box_volume**(1 / 2.)

    N = int(numpy.ceil(num_particles**(1. / 2.)))
    x = numpy.linspace(-L / 2, L / 2, N, endpoint=False)

    if x[1] - x[0] < 1.0:
        raise RuntimeError('density too high to initialize on square lattice')

    position_2d = list(itertools.product(x, repeat=2))[:num_particles]

    # create snapshot
    snap = gsd.hoomd.Snapshot()

    snap.particles.N = num_particles
    snap.particles.types = ['A']
    snap.configuration.box = [L, L, 0, 0, 0, 0]
    snap.particles.position = numpy.zeros(shape=(num_particles, 3))
    snap.particles.position[:, 0:2] = position_2d
    snap.particles.typeid = [0] * num_particles

    with gsd.hoomd.open(job.fn("hard_disk_initial_state.gsd"), 'wb') as f:
        f.append(snap)


# TODO: Refactor this and lj_fluid's make_simulation into utils.py
def make_simulation(job, device, initial_state, integrator, sim_mode, logger):
    """Make a simulation.

    Create a simulation, initialize its state, configure the table writer,
    trajectory writer, and quantity log file writer.

    Args:
        job (`signac.Job`): signac job object.

        device (`hoomd.device.Device`): hoomd device object.

        initial_state (str): Path to the gsd file to be used as an initial
            state.

        integrator (`hoomd.md.Integrator`): hoomd integrator object.

        sim_mode (str): String defining the simulation mode.

        logger (`hoomd.logging.Logger`): Logger object.
    """
    import hoomd

    suffix = 'cpu'
    if isinstance(device, hoomd.device.GPU):
        suffix = 'gpu'

    sim = hoomd.Simulation(device)
    sim.seed = job.statepoint.replicate_idx
    sim.create_state_from_gsd(initial_state)

    sim.operations.integrator = integrator

    # write to terminal
    logger_table = hoomd.logging.Logger(categories=['scalar'])
    logger_table.add(sim, quantities=['timestep', 'final_timestep', 'tps'])
    table_writer = hoomd.write.Table(hoomd.trigger.Periodic(WRITE_PERIOD),
                                     logger_table)
    sim.operations.add(table_writer)

    # write particle trajectory to gsd file
    trajectory_writer = hoomd.write.GSD(
        filename=job.fn(f"{sim_mode}_{suffix}_trajectory.gsd"),
        trigger=hoomd.trigger.Periodic(LOG_PERIOD['trajectory']),
        mode='wb')
    sim.operations.add(trajectory_writer)

    # write logged quantities to gsd file
    quantity_writer = hoomd.write.GSD(
        filter=hoomd.filter.Null(),
        filename=job.fn(f"{sim_mode}_{suffix}_quantities.gsd"),
        trigger=hoomd.trigger.Periodic(LOG_PERIOD['quantities']),
        mode='wb',
        log=logger)
    sim.operations.add(quantity_writer)

    return sim


def make_mc_simulation(job,
                       device,
                       initial_state,
                       sim_mode,
                       extra_loggables=[]):
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
    import hoomd

    # integrator
    mc = hoomd.hpmc.integrate.Sphere(nselect=4)
    mc.shape['A'] = dict(diameter=1.0)

    # log to gsd
    logger_gsd = hoomd.logging.Logger(categories=['scalar', 'sequence'])
    logger_gsd.add(mc, quantities=['translate_moves'])
    for loggable, quantity in extra_loggables:
        logger_gsd.add(loggable, quantities=[quantity])

    # make simulation
    sim = make_simulation(job, device, initial_state, mc, sim_mode, logger_gsd)

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
        trigger=hoomd.trigger.And([
            hoomd.trigger.Periodic(100),
            hoomd.trigger.Before(sim.timestep + int(RANDOMIZE_STEPS))
        ]))
    sim.operations.add(move_size_tuner)

    return sim


def run_nvt_sim(job, device):
    """Run MC sim in NVT."""
    import hoomd
    initial_state = job.fn('hard_disk_initial_state.gsd')
    sim_mode = 'nvt'

    sdf = hoomd.hpmc.compute.SDF(xmax=0.02, dx=1e-4)

    sim = make_mc_simulation(job,
                             device,
                             initial_state,
                             sim_mode,
                             extra_loggables=[(sdf, 'betaP')])

    sim.operations.computes.append(sdf)

    # equilibrate
    device.notice('Equilibrating...')
    sim.run(RANDOMIZE_STEPS)
    sim.run(RANDOMIZE_STEPS)
    device.notice('Done.')

    # Print acceptance ratio
    translate_moves = sim.operations.integrator.translate_moves
    translate_acceptance = translate_moves[0] / sum(translate_moves)
    device.notice(f'Translate move acceptance: {translate_acceptance}')
    device.notice(f'Trial move size: {sim.operations.integrator.d["A"]}')

    # run
    device.notice('Running...')
    sim.run(RUN_STEPS)
    device.notice('Done.')


@Project.operation(directives=dict(walltime=12,
                                   executable=CONFIG["executable"],
                                   nranks=16))
@Project.pre.after(hard_disk_create_initial_state)
@Project.post.true('hard_disk_nvt_cpu_complete')
def hard_disk_nvt_cpu(job):
    """Run NVT on the CPU."""
    import hoomd
    device = hoomd.device.CPU(msg_file=job.fn('run_nvt_cpu.log'))
    run_nvt_sim(job, device)

    if device.communicator.rank == 0:
        job.document['hard_disk_nvt_cpu_complete'] = True


@Project.operation(directives=dict(walltime=12,
                                   executable=CONFIG["executable"],
                                   nranks=1,
                                   ngpu=1))
@Project.pre.after(hard_disk_create_initial_state)
@Project.post.true('hard_disk_nvt_gpu_complete')
def hard_disk_nvt_gpu(job):
    """Run NVT on the GPU."""
    import hoomd
    device = hoomd.device.GPU(msg_file=job.fn('run_nvt_gpu.log'))
    run_nvt_sim(job, device)

    if device.communicator.rank == 0:
        job.document['hard_disk_nvt_gpu_complete'] = True


def run_npt_sim(job, device):
    """Run MC sim in NPT."""
    import hoomd
    from custom_actions import ComputeDensity

    # device
    initial_state = job.fn('hard_disk_initial_state.gsd')
    sim_mode = 'npt'

    # compute the density
    compute_density = ComputeDensity()

    # box updates
    boxmc = hoomd.hpmc.update.BoxMC(betaP=job.statepoint.pressure,
                                    trigger=hoomd.trigger.Periodic(1))
    boxmc.volume = dict(weight=1.0, mode='ln', delta=0.001)

    # simulation
    sim = make_mc_simulation(job,
                             device,
                             initial_state,
                             sim_mode,
                             extra_loggables=[
                                 (compute_density, 'density'),
                                 (boxmc, 'volume_moves'),
                             ])

    sim.operations.add(boxmc)

    boxmc_tuner = hoomd.hpmc.tune.BoxMCMoveSize.scale_solver(
        trigger=hoomd.trigger.And([
            hoomd.trigger.Periodic(200),
            hoomd.trigger.Before(sim.timestep + int(RANDOMIZE_STEPS))
        ]),
        boxmc=boxmc,
        moves=['volume'],
        target=0.5)
    sim.operations.add(boxmc_tuner)

    # equilibrate
    device.notice('Equilibrating...')
    sim.run(RANDOMIZE_STEPS)
    sim.run(RANDOMIZE_STEPS)
    device.notice('Done.')

    # Print acceptance ratio
    translate_moves = sim.operations.integrator.translate_moves
    translate_acceptance = translate_moves[0] / sum(translate_moves)
    device.notice(f'Translate move acceptance: {translate_acceptance}')
    device.notice(f'Trial move size: {sim.operations.integrator.d["A"]}')

    volume_moves = boxmc.volume_moves
    volume_acceptance = volume_moves[0] / sum(volume_moves)
    device.notice(f'Volume move acceptance: {volume_acceptance}')
    device.notice(f'Volume move size: {boxmc.volume["delta"]}')

    # run
    device.notice('Running...')
    sim.run(RUN_STEPS)
    device.notice('Done.')


@Project.operation(directives=dict(walltime=12,
                                   executable=CONFIG["executable"],
                                   nranks=16))
@Project.pre.after(hard_disk_create_initial_state)
@Project.post.true('hard_disk_npt_cpu_complete')
def hard_disk_npt_cpu(job):
    """Run NPT MC on the CPU."""
    import hoomd
    device = hoomd.device.CPU(msg_file=job.fn('run_npt_cpu.log'))
    run_npt_sim(job, device)

    if device.communicator.rank == 0:
        job.document['hard_disk_npt_cpu_complete'] = True


def run_nec_sim(job, device):
    """Run MC sim in NVT with NEC."""
    import hoomd
    initial_state = job.fn('hard_disk_initial_state.gsd')
    sim_mode = 'nec'

    mc = hoomd.hpmc.nec.integrate.Sphere(default_d=0.05, update_fraction=0.01, nselect=1)
    mc.shape['A'] = dict(diameter=1)
    mc.chain_time = 0.05

    sdf = hoomd.hpmc.compute.SDF(xmax=0.02, dx=1e-4)

    # log to gsd
    logger_gsd = hoomd.logging.Logger(categories=['scalar', 'sequence'])
    logger_gsd.add(mc, quantities=['translate_moves', 'particles_per_chain', 'virial_pressure'])
    logger_gsd.add(sdf, quantities=['betaP'])

    # make simulation
    sim = make_simulation(job, device, initial_state, mc, sim_mode, logger_gsd)

    sim.operations.computes.append(sdf)

    trigger_tune = hoomd.trigger.And([
            hoomd.trigger.Periodic(5),
            hoomd.trigger.Before(sim.timestep + int(RANDOMIZE_STEPS))])

    tune_nec_d = hoomd.hpmc.tune.MoveSize.scale_solver(
        trigger=trigger_tune,
        moves=['d'],
        target=0.10,
        tol=0.001,
        max_translation_move=0.25)
    sim.operations.tuners.append(tune_nec_d)

    tune_nec_ct = hoomd.hpmc.nec.tune.ChainTime.scale_solver(trigger_tune,
                                                             target=20.0,
                                                             tol=1.0,
                                                             gamma=20.0)
    sim.operations.tuners.append(tune_nec_ct)

    # equilibrate
    sim.state.thermalize_particle_momenta(hoomd.filter.All(), kT=1)

    device.notice('Equilibrating...')
    sim.run(RANDOMIZE_STEPS)
    sim.run(RANDOMIZE_STEPS)
    device.notice('Done.')

    # Print acceptance ratio
    translate_moves = sim.operations.integrator.translate_moves
    translate_acceptance = translate_moves[0] / sum(translate_moves)
    device.notice(f'Collision search acceptance: {translate_acceptance}')
    device.notice(f'Collision search size: {sim.operations.integrator.d["A"]}')
    device.notice(f'Particles per chain: {mc.particles_per_chain}')

    # run
    device.notice('Running...')
    sim.run(RUN_STEPS)
    device.notice('Done.')


@Project.operation(directives=dict(walltime=48,
                                   executable=CONFIG["executable"],
                                   nranks=1))
@Project.pre.after(hard_disk_create_initial_state)
@Project.post.true('hard_disk_nec_cpu_complete')
def hard_disk_nec_cpu(job):
    """Run NEC on the CPU."""
    import hoomd
    device = hoomd.device.CPU(msg_file=job.fn('run_nec_cpu.log'))
    run_nec_sim(job, device)

    if device.communicator.rank == 0:
        job.document['hard_disk_nec_cpu_complete'] = True


@Project.operation(directives=dict(walltime=1, executable=CONFIG["executable"]))
@Project.pre.after(hard_disk_nvt_cpu)
@Project.pre.after(hard_disk_nvt_gpu)
@Project.pre.after(hard_disk_npt_cpu)
@Project.post.true('hard_disk_analysis_complete')
def hard_disk_analyze(job):
    """Analyze the output of all simulation modes."""
    import gsd.hoomd
    import numpy
    import matplotlib
    import matplotlib.style
    import matplotlib.figure
    matplotlib.style.use('ggplot')
    from util import read_gsd_log_trajectory, get_log_quantity

    constant = dict(nvt_cpu='density', nvt_gpu='density', npt_cpu='pressure')
    sim_modes = [
        'nvt_cpu',
        'nvt_gpu',
        'npt_cpu',
    ]

    pressures = {}
    densities = {}

    for sim_mode in sim_modes:
        with gsd.hoomd.open(job.fn(sim_mode + '_quantities.gsd')) as gsd_traj:
            # read GSD file once
            traj = read_gsd_log_trajectory(gsd_traj)

        n_frames = len(traj)

        if constant[sim_mode] == 'density':
            pressures[sim_mode] = get_log_quantity(traj,
                                                   'hpmc/compute/SDF/betaP')
        else:
            pressures[sim_mode] = numpy.ones(n_frames) * job.statepoint.pressure

        if constant[sim_mode] == 'pressure':
            densities[sim_mode] = get_log_quantity(
                traj, 'custom_actions/ComputeDensity/density')
        else:
            densities[sim_mode] = numpy.ones(n_frames) * job.statepoint.density

    # save averages
    for mode in sim_modes:
        job.document[mode] = dict(
            pressure=float(numpy.mean(pressures[mode][-FRAMES_ANALYZE:])),
            density=float(numpy.mean(densities[mode][-FRAMES_ANALYZE:])))

    # Plot results
    fig = matplotlib.figure.Figure(figsize=(10, 10 / 1.618 * 3), layout='tight')
    ax = fig.add_subplot(3, 1, 1)

    # subsample the values for time series plots
    sample_rate = 8
    for mode in sim_modes:
        ax.plot(densities[mode][::sample_rate], label=mode)
        ax.set_xlabel('frame')
        ax.set_ylabel(r'$\rho$')
        ax.legend()

    ax.hlines(y=job.statepoint.density,
              xmin=0,
              xmax=len(densities[sim_modes[0]]) / sample_rate,
              linestyles='dashed',
              colors='k')

    ax = fig.add_subplot(3, 1, 2)
    for mode in sim_modes:
        ax.plot(pressures[mode][::sample_rate], label=mode)
        ax.set_xlabel('frame')
        ax.set_ylabel('$P$')

    ax.hlines(y=job.statepoint.pressure,
              xmin=0,
              xmax=len(densities[sim_modes[0]]) / sample_rate,
              linestyles='dashed',
              colors='k')

    # determine range for density and pressure histograms
    density_range = [
        numpy.min(densities[sim_modes[0]][-FRAMES_ANALYZE:]),
        numpy.max(densities[sim_modes[0]][-FRAMES_ANALYZE:])
    ]
    pressure_range = [
        numpy.min(pressures[sim_modes[0]][-FRAMES_ANALYZE:]),
        numpy.max(pressures[sim_modes[0]][-FRAMES_ANALYZE:])
    ]

    for mode in sim_modes[1:]:
        density_range[0] = min(density_range[0],
                               numpy.min(densities[mode][-FRAMES_ANALYZE:]))
        density_range[1] = max(density_range[1],
                               numpy.max(densities[mode][-FRAMES_ANALYZE:]))
        pressure_range[0] = min(pressure_range[0],
                                numpy.min(pressures[mode][-FRAMES_ANALYZE:]))
        pressure_range[1] = max(pressure_range[1],
                                numpy.max(pressures[mode][-FRAMES_ANALYZE:]))

    ax = fig.add_subplot(3, 2, 5)
    max_density_histogram = 0
    for mode in sim_modes:
        density_histogram, bin_edges = numpy.histogram(
            densities[mode][-FRAMES_ANALYZE:], bins=50, range=density_range)
        if constant[mode] == 'density':
            density_histogram[:] = 0

        max_density_histogram = max(max_density_histogram,
                                    numpy.max(density_histogram))

        ax.plot(bin_edges[:-1], density_histogram, label=mode)
        ax.set_xlabel(r'$\rho$')
        ax.set_ylabel('frequency')

    ax.vlines(x=job.statepoint.density,
              ymin=0,
              ymax=max_density_histogram,
              linestyles='dashed',
              colors='k')

    ax = fig.add_subplot(3, 2, 6)
    max_pressure_histogram = 0
    for mode in sim_modes:
        pressure_histogram, bin_edges = numpy.histogram(
            pressures[mode][-FRAMES_ANALYZE:], bins=50, range=pressure_range)
        if constant[mode] == 'pressure':
            pressure_histogram[:] = 0

        max_pressure_histogram = max(max_pressure_histogram,
                                     numpy.max(pressure_histogram))

        ax.plot(bin_edges[:-1], pressure_histogram, label=mode)
        ax.set_xlabel(r'$P$')
        ax.set_ylabel('frequency')

    ax.vlines(x=job.statepoint.pressure,
              ymin=0,
              ymax=max_pressure_histogram,
              linestyles='dashed',
              colors='k')

    fig.suptitle(f"$\\rho={job.statepoint.density}$, "
                 f"$N={job.statepoint.num_particles}$, "
                 f"replicate={job.statepoint.replicate_idx}")
    fig.savefig(job.fn('nvt_npt_plots.svg'), bbox_inches='tight')

    job.document['hard_disk_analysis_complete'] = True


@aggregator.groupby(key=['density', 'num_particles'],
                    sort_by='replicate_idx',
                    select=is_hard_disk)
@Project.operation(directives=dict(executable=CONFIG["executable"]))
@Project.pre(
    lambda *jobs: util.true_all(*jobs, key='hard_disk_analysis_complete'))
@Project.post(
    lambda *jobs: util.true_all(*jobs, key='hard_disk_compare_modes_complete'))
def hard_disk_compare_modes(*jobs):
    """Compares the tested simulation modes."""
    import numpy
    import matplotlib
    import matplotlib.style
    import matplotlib.figure
    import scipy.stats
    matplotlib.style.use('ggplot')

    sim_modes = [
        'nvt_cpu',
        'nvt_gpu',
        'npt_cpu',
    ]
    quantity_names = ['density', 'pressure']

    # grab the common statepoint parameters
    set_density = jobs[0].sp.density
    set_pressure = jobs[0].sp.pressure
    num_particles = jobs[0].sp.num_particles

    quantity_reference = dict(density=set_density,
                              pressure=set_pressure,
                              potential_energy=None)

    fig = matplotlib.figure.Figure(figsize=(10, 10 / 1.618 * 2), layout='tight')
    fig.suptitle(f"$\\rho={set_density}$, $N={num_particles}$")

    for i, quantity_name in enumerate(quantity_names):
        ax = fig.add_subplot(2, 1, i + 1)

        # organize data from jobs
        quantities = {mode: [] for mode in sim_modes}
        for jb in jobs:
            for mode in sim_modes:
                quantities[mode].append(
                    getattr(getattr(jb.doc, mode), quantity_name))

        # compute stats with data
        avg_quantity = {
            mode: numpy.mean(quantities[mode]) for mode in sim_modes
        }
        stderr_quantity = {
            mode:
            2 * numpy.std(quantities[mode]) / numpy.sqrt(len(quantities[mode]))
            for mode in sim_modes
        }

        # compute the quantity differences
        quantity_list = [avg_quantity[mode] for mode in sim_modes]
        stderr_list = numpy.array([stderr_quantity[mode] for mode in sim_modes])

        if quantity_reference[quantity_name] is not None:
            reference = quantity_reference[quantity_name]
        else:
            reference = numpy.mean(quantity_list)

        quantity_diff_list = numpy.array(quantity_list) - reference

        ax.errorbar(x=range(len(sim_modes)),
                    y=quantity_diff_list / reference / 1e-3,
                    yerr=numpy.fabs(stderr_list / reference / 1e-3),
                    fmt='s')
        ax.set_xticks(range(len(sim_modes)), sim_modes, rotation=45)
        ax.set_ylabel(quantity_name + ' relative error / 1e-3')
        ax.hlines(y=0,
                  xmin=0,
                  xmax=len(sim_modes) - 1,
                  linestyles='dashed',
                  colors='k')

        unpacked_quantities = list(quantities.values())
        f, p = scipy.stats.f_oneway(*unpacked_quantities)

        if p > 0.05:
            result = "$\\checkmark$"
        else:
            result = "XX"

        ax.set_title(label=result + f' ANOVA p-value: {p:0.3f}')

    filename = f'hard_disk_compare_density{round(set_density, 2)}.svg'
    fig.savefig(os.path.join(jobs[0]._project.path, filename),
                bbox_inches='tight')

    for job in jobs:
        job.document['hard_disk_compare_modes_complete'] = True
