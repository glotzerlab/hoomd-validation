# Copyright (c) 2022 The Regents of the University of Michigan.
# Part of HOOMD-blue, released under the BSD 3-Clause License.

"""Hard sphere equation of state validation test."""

from config import CONFIG
from project_class import Project
from flow import aggregator
import util
import os
import math

# Run parameters shared between simulations.
# Step counts must be even and a multiple of the log quantity period.
RANDOMIZE_STEPS = 20_000
EQUILIBRATE_STEPS = 100_000
RUN_STEPS = 1_000_000
TOTAL_STEPS = RANDOMIZE_STEPS + EQUILIBRATE_STEPS + RUN_STEPS

WRITE_PERIOD = 1_000
LOG_PERIOD = {'trajectory': 50_000, 'quantities': 1_000}
NUM_CPU_RANKS = min(8, CONFIG["max_cores_sim"])


def job_statepoints():
    """list(dict): A list of statepoints for this subproject."""
    num_particles = 11**3
    replicate_indices = range(CONFIG["replicates"])
    # Reference statepoint(s) from: https://doi.org/10.1002/aic.10233 .
    params_list = [(0.75, 6.6566), (0.6, 4.2952)]
    for density, compressibility in params_list:
        for idx in replicate_indices:
            yield ({
                "subproject": "hard_sphere",
                "density": density,
                "pressure": compressibility * density,
                "num_particles": num_particles,
                "replicate_idx": idx
            })


def is_hard_sphere(job):
    """Test if a given job is part of the hard_sphere subproject."""
    return job.statepoint['subproject'] == 'hard_sphere'


partition_jobs_cpu_serial = aggregator.groupsof(num=min(
    CONFIG["replicates"], CONFIG["max_cores_submission"]),
                                                sort_by='density',
                                                select=is_hard_sphere)

partition_jobs_cpu_mpi = aggregator.groupsof(num=min(
    CONFIG["replicates"], CONFIG["max_cores_submission"] // NUM_CPU_RANKS),
                                             sort_by='density',
                                             select=is_hard_sphere)

partition_jobs_gpu = aggregator.groupsof(num=min(CONFIG["replicates"],
                                                 CONFIG["max_gpus_submission"]),
                                         sort_by='density',
                                         select=is_hard_sphere)


@Project.post.isfile('hard_sphere_initial_state.gsd')
@Project.operation(directives=dict(
    executable=CONFIG["executable"],
    nranks=util.total_ranks_function(NUM_CPU_RANKS),
    walltime=1),
                   aggregator=partition_jobs_cpu_mpi)
def hard_sphere_create_initial_state(*jobs):
    """Create initial system configuration."""
    import hoomd
    import numpy
    import itertools

    communicator = hoomd.communicator.Communicator(
        ranks_per_partition=NUM_CPU_RANKS)
    job = jobs[communicator.partition]

    if communicator.rank == 0:
        print('starting hard_sphere_create_initial_state:', job)

    num_particles = job.statepoint['num_particles']
    density = job.statepoint['density']

    box_volume = num_particles / density
    L = box_volume**(1 / 3.)

    N = int(numpy.ceil(num_particles**(1. / 3.)))
    x = numpy.linspace(-L / 2, L / 2, N, endpoint=False)

    if x[1] - x[0] < 1.0:
        raise RuntimeError('density too high to initialize on square lattice')

    position = list(itertools.product(x, repeat=3))[:num_particles]

    # create snapshot
    device = hoomd.device.CPU(communicator=communicator,
                              msg_file=job.fn('create_initial_state.log'))
    snap = hoomd.Snapshot(device.communicator)

    if device.communicator.rank == 0:
        snap.particles.N = num_particles
        snap.particles.types = ['A']
        snap.configuration.box = [L, L, L, 0, 0, 0]
        snap.particles.position[:] = position
        snap.particles.typeid[:] = [0] * num_particles

    # Use hard sphere Monte-Carlo to randomize the initial configuration
    mc = hoomd.hpmc.integrate.Sphere(default_d=0.01)
    mc.shape['A'] = dict(diameter=1.0)

    sim = hoomd.Simulation(device=device, seed=job.statepoint.replicate_idx)
    sim.create_state_from_snapshot(snap)
    sim.operations.integrator = mc

    device.notice('Randomizing initial state...')
    sim.run(RANDOMIZE_STEPS)
    device.notice(f'Done. Move counts: {mc.translate_moves}')

    hoomd.write.GSD.write(state=sim.state,
                          filename=job.fn("hard_sphere_initial_state.gsd"),
                          mode='wb')


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
    sim = util.make_simulation(job=job,
                               device=device,
                               initial_state=initial_state,
                               integrator=mc,
                               sim_mode=sim_mode,
                               logger=logger_gsd,
                               table_write_period=WRITE_PERIOD,
                               trajectory_write_period=LOG_PERIOD['trajectory'],
                               log_write_period=LOG_PERIOD['quantities'],
                               log_start_step=RANDOMIZE_STEPS
                               + EQUILIBRATE_STEPS)

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
            hoomd.trigger.Before(RANDOMIZE_STEPS + EQUILIBRATE_STEPS // 2)
        ]))
    sim.operations.add(move_size_tuner)

    return sim


def run_nvt_sim(job, device):
    """Run MC sim in NVT."""
    import hoomd
    initial_state = job.fn('hard_sphere_initial_state.gsd')
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
    sim.run(EQUILIBRATE_STEPS // 2)
    sim.run(EQUILIBRATE_STEPS // 2)
    device.notice('Done.')

    # Print acceptance ratio as measured during the 2nd half of the
    # equilibration.
    translate_moves = sim.operations.integrator.translate_moves
    translate_acceptance = translate_moves[0] / sum(translate_moves)
    device.notice(f'Translate move acceptance: {translate_acceptance}')
    device.notice(f'Trial move size: {sim.operations.integrator.d["A"]}')

    # run
    device.notice('Running...')
    sim.run(RUN_STEPS)
    device.notice('Done.')


def run_npt_sim(job, device):
    """Run MC sim in NPT."""
    import hoomd
    from custom_actions import ComputeDensity

    # device
    initial_state = job.fn('hard_sphere_initial_state.gsd')
    sim_mode = 'npt'

    # compute the density
    compute_density = ComputeDensity()

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
                                 (compute_density, 'density'),
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

    # equilibrate
    device.notice('Equilibrating...')
    sim.run(EQUILIBRATE_STEPS // 2)
    sim.run(EQUILIBRATE_STEPS // 2)
    device.notice('Done.')

    # Print acceptance ratio as measured during the 2nd half of the
    # equilibration
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


def run_nec_sim(job, device):
    """Run MC sim in NVT with NEC."""
    import hoomd
    initial_state = job.fn('hard_sphere_initial_state.gsd')
    sim_mode = 'nec'

    mc = hoomd.hpmc.nec.integrate.Sphere(default_d=0.05,
                                         update_fraction=0.01,
                                         nselect=1)
    mc.shape['A'] = dict(diameter=1)
    mc.chain_time = 0.05

    sdf = hoomd.hpmc.compute.SDF(xmax=0.02, dx=1e-4)

    # log to gsd
    logger_gsd = hoomd.logging.Logger(categories=['scalar', 'sequence'])
    logger_gsd.add(mc,
                   quantities=[
                       'translate_moves', 'particles_per_chain',
                       'virial_pressure'
                   ])
    logger_gsd.add(sdf, quantities=['betaP'])

    # make simulation
    sim = util.make_simulation(job=job,
                               device=device,
                               initial_state=initial_state,
                               integrator=mc,
                               sim_mode=sim_mode,
                               logger=logger_gsd,
                               table_write_period=WRITE_PERIOD,
                               trajectory_write_period=LOG_PERIOD['trajectory'],
                               log_write_period=LOG_PERIOD['quantities'],
                               log_start_step=RANDOMIZE_STEPS
                               + EQUILIBRATE_STEPS)

    sim.operations.computes.append(sdf)

    trigger_tune = hoomd.trigger.And([
        hoomd.trigger.Periodic(5),
        hoomd.trigger.Before(RANDOMIZE_STEPS + EQUILIBRATE_STEPS // 2)
    ])

    tune_nec_d = hoomd.hpmc.tune.MoveSize.scale_solver(trigger=trigger_tune,
                                                       moves=['d'],
                                                       target=0.10,
                                                       tol=0.001,
                                                       max_translation_move=1.0)
    sim.operations.tuners.append(tune_nec_d)

    tune_nec_ct = hoomd.hpmc.nec.tune.ChainTime.scale_solver(trigger_tune,
                                                             target=20.0,
                                                             tol=1.0,
                                                             gamma=20.0)
    sim.operations.tuners.append(tune_nec_ct)

    # equilibrate
    sim.state.thermalize_particle_momenta(hoomd.filter.All(), kT=1)

    device.notice('Equilibrating...')
    sim.run(EQUILIBRATE_STEPS // 2)
    sim.run(EQUILIBRATE_STEPS // 2)
    device.notice('Done.')

    # Print acceptance ratio as measured during the 2nd half of the
    # equilibration
    translate_moves = sim.operations.integrator.translate_moves
    if sum(translate_moves) > 0:
        translate_acceptance = translate_moves[0] / sum(translate_moves)
        device.notice(f'Collision search acceptance: {translate_acceptance}')
    else:
        raise RuntimeError('No translate moves!')

    device.notice(f'Collision search size: {sim.operations.integrator.d["A"]}')
    device.notice(f'Particles per chain: {mc.particles_per_chain}')

    # run
    device.notice('Running...')
    sim.run(RUN_STEPS)
    device.notice('Done.')


sampling_jobs = []
job_definitions = [
    {
        'mode': 'nvt',
        'device_name': 'cpu',
        'ranks_per_partition': NUM_CPU_RANKS,
        'aggregator': partition_jobs_cpu_mpi
    },
    {
        'mode': 'nvt',
        'device_name': 'gpu',
        'ranks_per_partition': 1,
        'aggregator': partition_jobs_gpu
    },
    {
        'mode': 'npt',
        'device_name': 'cpu',
        'ranks_per_partition': NUM_CPU_RANKS,
        'aggregator': partition_jobs_cpu_mpi
    },
    {
        'mode': 'nec',
        'device_name': 'cpu',
        'ranks_per_partition': 1,
        'aggregator': partition_jobs_cpu_serial
    },
]


def add_sampling_job(mode, device_name, ranks_per_partition, aggregator):
    """Add a sampling job to the workflow."""
    directives = dict(walltime=CONFIG["max_walltime"],
                      executable=CONFIG["executable"],
                      nranks=util.total_ranks_function(ranks_per_partition))

    if device_name == 'gpu':
        directives['ngpu'] = directives['nranks']

    @Project.pre.after(hard_sphere_create_initial_state)
    @Project.post(
        util.gsd_step_greater_equal_function(
            f'{mode}_{device_name}_quantities.gsd', TOTAL_STEPS))
    @Project.operation(name=f'hard_sphere_{mode}_{device_name}',
                       directives=directives,
                       aggregator=aggregator)
    def sampling_operation(*jobs):
        """Perform sampling simulation given the definition."""
        import hoomd

        communicator = hoomd.communicator.Communicator(
            ranks_per_partition=ranks_per_partition)
        job = jobs[communicator.partition]

        if communicator.rank == 0:
            print(f'starting hard_sphere_{mode}_{device_name}', job)

        if device_name == 'gpu':
            device_cls = hoomd.device.GPU
        elif device_name == 'cpu':
            device_cls = hoomd.device.CPU

        device = device_cls(communicator=communicator,
                            msg_file=job.fn(f'run_{mode}_{device_name}.log'))

        globals().get(f'run_{mode}_sim')(job, device)

    sampling_jobs.append(sampling_operation)


for definition in job_definitions:
    add_sampling_job(**definition)


@Project.pre.after(*sampling_jobs)
@Project.post.true('hard_sphere_analysis_complete')
@Project.operation(directives=dict(walltime=1, executable=CONFIG["executable"]))
def hard_sphere_analyze(job):
    """Analyze the output of all simulation modes."""
    import gsd.hoomd
    import numpy
    import matplotlib
    import matplotlib.style
    import matplotlib.figure
    matplotlib.style.use('ggplot')
    from util import read_gsd_log_trajectory, get_log_quantity

    print('starting hard_sphere_analyze:', job)

    constant = dict(nvt_cpu='density',
                    nvt_gpu='density',
                    nec_cpu='density',
                    npt_cpu='pressure')
    sim_modes = list(constant.keys())

    pressures = {}
    densities = {}

    for sim_mode in sim_modes:
        with gsd.hoomd.open(job.fn(sim_mode + '_quantities.gsd')) as gsd_traj:
            # read GSD file once
            traj = read_gsd_log_trajectory(gsd_traj)

        n_frames = len(traj)

        # NEC generates inf virial pressures for 2D simulations in HOOMD-blue
        # v3.6.0, fall back on SDF
        if constant[sim_mode] == 'density' and (
                'nvt' in sim_mode
                or traj[0]['hpmc/nec/integrate/Sphere/virial_pressure'][0]
                == math.inf):
            pressures[sim_mode] = get_log_quantity(traj,
                                                   'hpmc/compute/SDF/betaP')
        elif constant[sim_mode] == 'density' and 'nec' in sim_mode:
            pressures[sim_mode] = get_log_quantity(
                traj, 'hpmc/nec/integrate/Sphere/virial_pressure')
        else:
            pressures[sim_mode] = numpy.ones(n_frames) * job.statepoint.pressure

        if constant[sim_mode] == 'pressure':
            densities[sim_mode] = get_log_quantity(
                traj, 'custom_actions/ComputeDensity/density')
        else:
            densities[sim_mode] = numpy.ones(n_frames) * job.statepoint.density

    # save averages
    for mode in sim_modes:
        job.document[mode] = dict(pressure=float(numpy.mean(pressures[mode])),
                                  density=float(numpy.mean(densities[mode])))

    # Plot results
    def plot(*, ax, data, quantity_name, base_line=None, legend=False):
        # subsample the values for time series plots
        for mode in sim_modes:
            ax.plot(data[mode], label=mode)
        ax.set_xlabel('frame')
        ax.set_ylabel(quantity_name)

        if legend:
            ax.legend()

        if base_line is not None:
            ax.hlines(y=base_line,
                      xmin=0,
                      xmax=len(data[sim_modes[0]]),
                      linestyles='dashed',
                      colors='k')

    fig = matplotlib.figure.Figure(figsize=(10, 10 / 1.618 * 3), layout='tight')
    ax = fig.add_subplot(3, 1, 1)

    plot(ax=ax,
         data=densities,
         quantity_name=r"$\rho",
         base_line=job.sp.density,
         legend=True)

    ax = fig.add_subplot(3, 1, 2)
    plot(ax=ax, data=pressures, quantity_name=r"$P", base_line=job.sp.pressure)

    # determine range for density and pressure histograms
    density_range = [
        numpy.min(densities[sim_modes[0]]),
        numpy.max(densities[sim_modes[0]])
    ]
    pressure_range = [
        numpy.min(pressures[sim_modes[0]]),
        numpy.max(pressures[sim_modes[0]])
    ]

    for mode in sim_modes[1:]:
        density_range[0] = min(density_range[0], numpy.min(densities[mode]))
        density_range[1] = max(density_range[1], numpy.max(densities[mode]))
        pressure_range[0] = min(pressure_range[0], numpy.min(pressures[mode]))
        pressure_range[1] = max(pressure_range[1], numpy.max(pressures[mode]))

    def plot_histogram(*, ax, data, quantity_name, sp_name, range):
        max_histogram = 0
        for mode in sim_modes:
            histogram, bin_edges = numpy.histogram(data[mode],
                                                   bins=50,
                                                   range=range)
            if constant[mode] == sp_name:
                histogram[:] = 0

            max_histogram = max(max_histogram, numpy.max(histogram))

            ax.plot(bin_edges[:-1], histogram, label=mode)
        ax.set_xlabel(quantity_name)
        ax.set_ylabel('frequency')
        ax.vlines(x=job.sp[sp_name],
                  ymin=0,
                  ymax=max_histogram,
                  linestyles='dashed',
                  colors='k')

    ax = fig.add_subplot(3, 2, 5)
    plot_histogram(ax=ax,
                   data=densities,
                   quantity_name=r"$\rho$",
                   sp_name="density",
                   range=density_range)

    ax = fig.add_subplot(3, 2, 6)
    plot_histogram(ax=ax,
                   data=pressures,
                   quantity_name="$P$",
                   sp_name="pressure",
                   range=pressure_range)

    fig.suptitle(f"$\\rho={job.statepoint.density}$, "
                 f"$N={job.statepoint.num_particles}$, "
                 f"replicate={job.statepoint.replicate_idx}")
    fig.savefig(job.fn('nvt_npt_plots.svg'), bbox_inches='tight')

    job.document['hard_sphere_analysis_complete'] = True


@Project.pre(
    lambda *jobs: util.true_all(*jobs, key='hard_sphere_analysis_complete'))
@Project.post(
    lambda *jobs: util.true_all(*jobs, key='hard_sphere_compare_modes_complete')
)
@Project.operation(directives=dict(executable=CONFIG["executable"]),
                   aggregator=aggregator.groupby(
                       key=['density', 'num_particles'],
                       sort_by='replicate_idx',
                       select=is_hard_sphere))
def hard_sphere_compare_modes(*jobs):
    """Compares the tested simulation modes."""
    import numpy
    import matplotlib
    import matplotlib.style
    import matplotlib.figure
    import scipy.stats
    matplotlib.style.use('ggplot')

    print('starting hard_sphere_compare_modes:', jobs[0])

    sim_modes = [
        'nvt_cpu',
        'nvt_gpu',
        'nec_cpu',
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

    filename = f'hard_sphere_compare_density{round(set_density, 2)}.svg'
    fig.savefig(os.path.join(jobs[0]._project.path, filename),
                bbox_inches='tight')

    for job in jobs:
        job.document['hard_sphere_compare_modes_complete'] = True
