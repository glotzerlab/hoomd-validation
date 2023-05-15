# Copyright (c) 2022-2023 The Regents of the University of Michigan.
# Part of HOOMD-blue, released under the BSD 3-Clause License.

"""Lennard Jones phase behavior validation test."""

from config import CONFIG
from project_class import Project
from flow import aggregator
import util
import os
import math
import collections

# Run parameters shared between simulations.
# Step counts must be even and a multiple of the log quantity period.
RANDOMIZE_STEPS = 20_000
EQUILIBRATE_STEPS = 100_000
RUN_STEPS = 4_000_000
TOTAL_STEPS = RANDOMIZE_STEPS + EQUILIBRATE_STEPS + RUN_STEPS

WRITE_PERIOD = 4_000
LOG_PERIOD = {'trajectory': 50_000, 'quantities': 1000}
LJ_PARAMS = {'epsilon': 1.0, 'sigma': 1.0, 'r_on': 2.0}
NUM_CPU_RANKS = min(8, CONFIG["max_cores_sim"])

# Limit the number of long NVE runs to reduce the number of CPU hours needed.
NUM_NVE_RUNS = 2


def _sort_sim_modes(sim_modes):
    """Sort simulation modes for comparison."""
    sim_modes.sort(key=lambda x: ('nvt' in x, 'md' in x, x))


def job_statepoints():
    """list(dict): A list of statepoints for this subproject."""
    replicate_indices = range(CONFIG["replicates"])
    params_list = [
        dict(kT=1.5,
             density=0.5954153196609695,
             pressure=1.0,
             num_particles = 12**3,
             r_cut=2.5),
        dict(
            kT=1.0,
            density=0.9193740949934834,
            pressure=11.0,
            num_particles = 12**3,
            r_cut=2**(1 / 6),
        ),
    ]

    for param in params_list:
        for idx in replicate_indices:
            yield ({
                "subproject": "lj_fluid",
                "kT": param['kT'],
                "density": param['density'],
                "pressure": param['pressure'],
                "num_particles": param['num_particles'],
                "replicate_idx": idx,
                "r_cut": param['r_cut'],
            })


def is_lj_fluid(job):
    """Test if a given job is part of the lj_fluid subproject."""
    return job.statepoint['subproject'] == 'lj_fluid'


def sort_key(job):
    """Aggregator sort key."""
    return (job.statepoint.density, job.statepoint.num_particles)


partition_jobs_cpu_mpi = aggregator.groupsof(num=min(
    CONFIG["replicates"], CONFIG["max_cores_submission"] // NUM_CPU_RANKS),
                                             sort_by=sort_key,
                                             select=is_lj_fluid)

partition_jobs_gpu = aggregator.groupsof(num=min(CONFIG["replicates"],
                                                 CONFIG["max_gpus_submission"]),
                                         sort_by=sort_key,
                                         select=is_lj_fluid)


@Project.post.isfile('lj_fluid_initial_state.gsd')
@Project.operation(directives=dict(
    executable=CONFIG["executable"],
    nranks=util.total_ranks_function(NUM_CPU_RANKS),
    walltime=CONFIG['short_walltime']),
                   aggregator=partition_jobs_cpu_mpi)
def lj_fluid_create_initial_state(*jobs):
    """Create initial system configuration."""
    import hoomd
    import numpy
    import itertools

    communicator = hoomd.communicator.Communicator(
        ranks_per_partition=NUM_CPU_RANKS)
    job = jobs[communicator.partition]

    if communicator.rank == 0:
        print('starting lj_fluid_create_initial_state:', job)

    sp = job.sp
    device = hoomd.device.CPU(
        communicator=communicator,
        message_filename=job.fn('create_initial_state.log'))

    box_volume = sp["num_particles"] / sp["density"]
    L = box_volume**(1 / 3.)

    N = int(numpy.ceil(sp["num_particles"]**(1. / 3.)))
    x = numpy.linspace(-L / 2, L / 2, N, endpoint=False)

    if x[1] - x[0] < 1.0:
        raise RuntimeError('density too high to initialize on cubic lattice')

    position = list(itertools.product(x, repeat=3))[:sp["num_particles"]]

    # create snapshot
    snap = hoomd.Snapshot(device.communicator)

    if device.communicator.rank == 0:
        snap.particles.N = sp["num_particles"]
        snap.particles.types = ['A']
        snap.configuration.box = [L, L, L, 0, 0, 0]
        snap.particles.position[:] = position
        snap.particles.typeid[:] = [0] * sp["num_particles"]

    # Use hard sphere Monte-Carlo to randomize the initial configuration
    mc = hoomd.hpmc.integrate.Sphere()
    mc.shape['A'] = dict(diameter=1.0)

    sim = hoomd.Simulation(device=device, seed=util.make_seed(job))
    sim.create_state_from_snapshot(snap)
    sim.operations.integrator = mc

    device.notice('Randomizing initial state...')
    sim.run(RANDOMIZE_STEPS)
    device.notice(f'Done. Move counts: {mc.translate_moves}')

    hoomd.write.GSD.write(state=sim.state,
                          filename=job.fn("lj_fluid_initial_state.gsd"),
                          mode='wb')


#################################
# MD ensemble simulations
#################################


def make_md_simulation(job,
                       device,
                       initial_state,
                       method,
                       sim_mode,
                       extra_loggables=[],
                       period_multiplier=1):
    """Make an MD simulation.

    Args:
        job (`signac.job.Job`): Signac job object.

        device (`hoomd.device.Device`): hoomd device object.

        initial_state (str): Path to the gsd file to be used as an initial state
            for the simulation.

        method (`hoomd.md.methods.Method`): hoomd integration method.

        sim_mode (str): String identifying the simulation mode.

        extra_loggables (list): List of quantities to add to the gsd logger.

        ThermodynamicQuantities is added by default, any more quantities should
            be in this list.

        period_multiplier (int): Factor to multiply the GSD file periods by.
    """
    import hoomd
    from hoomd import md

    # pair force
    nlist = md.nlist.Cell(buffer=0.4)
    lj = md.pair.LJ(default_r_cut=job.statepoint.r_cut,
                    default_r_on=LJ_PARAMS['r_on'],
                    nlist=nlist)
    lj.params[('A', 'A')] = dict(sigma=LJ_PARAMS['sigma'],
                                 epsilon=LJ_PARAMS['epsilon'])
    lj.mode = 'xplor'

    # integrator
    integrator = md.Integrator(dt=0.0001, methods=[method], forces=[lj])

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
               ])
    logger.add(integrator, quantities=['linear_momentum'])
    for loggable in extra_loggables:
        logger.add(loggable)

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
        log_start_step=RANDOMIZE_STEPS + EQUILIBRATE_STEPS)
    sim.operations.add(thermo)
    for loggable in extra_loggables:
        # call attach explicitly so we can access sim state when computing the
        # loggable quantity
        if hasattr(loggable, 'attach'):
            loggable.attach(sim)

    return sim


def run_md_sim(job, device, ensemble, thermostat):
    """Run the MD simulation with the given ensemble and thermostat."""
    import hoomd
    from hoomd import md
    from custom_actions import ComputeDensity

    initial_state = job.fn('lj_fluid_initial_state.gsd')

    if ensemble == 'nvt':
        if thermostat == 'langevin':
            method = md.methods.Langevin(hoomd.filter.All(),
                                         kT=job.statepoint.kT)
            method.gamma.default = 1.0
        elif thermostat == 'mttk':
            method = md.methods.ConstantVolume(filter=hoomd.filter.All())
            method.thermostat = hoomd.md.methods.thermostats.MTTK(
                kT=job.statepoint.kT, tau=0.25)
        elif thermostat == 'bussi':
            method = md.methods.ConstantVolume(filter=hoomd.filter.All())
            method.thermostat = hoomd.md.methods.thermostats.Bussi(
                kT=job.statepoint.kT)
        else:
            raise ValueError(f'Unsupported thermostat {thermostat}')
    elif ensemble == 'npt':
        p = job.statepoint.pressure
        method = md.methods.ConstantPressure(hoomd.filter.All(),
                                             S=[p, p, p, 0, 0, 0],
                                             tauS=3,
                                             couple='xyz')
        if thermostat == 'bussi':
            method.thermostat = hoomd.md.methods.thermostats.Bussi(
                kT=job.statepoint.kT)
        else:
            raise ValueError(f'Unsupported thermostat {thermostat}')

    sim_mode = f'{ensemble}_{thermostat}_md'

    density_compute = ComputeDensity()
    sim = make_md_simulation(job,
                             device,
                             initial_state,
                             method,
                             sim_mode,
                             extra_loggables=[density_compute])

    # thermalize momenta
    sim.state.thermalize_particle_momenta(hoomd.filter.All(), job.sp.kT)

    # thermalize the thermostat (if applicable)
    if ((isinstance(method, md.methods.ConstantVolume)
         or isinstance(method, md.methods.ConstantPressure))
            and hasattr(method.thermostat, 'thermalize_dof')):
        sim.run(0)
        method.thermostat.thermalize_dof()

    # equilibrate
    device.notice('Equilibrating...')
    sim.run(EQUILIBRATE_STEPS)
    device.notice('Done.')

    # run
    device.notice('Running...')
    sim.run(RUN_STEPS)
    device.notice('Done.')


md_sampling_jobs = []
md_job_definitions = [
    {
        'ensemble': 'nvt',
        'thermostat': 'langevin',
        'device_name': 'cpu',
        'ranks_per_partition': NUM_CPU_RANKS,
        'aggregator': partition_jobs_cpu_mpi
    },
    {
        'ensemble': 'nvt',
        'thermostat': 'mttk',
        'device_name': 'cpu',
        'ranks_per_partition': NUM_CPU_RANKS,
        'aggregator': partition_jobs_cpu_mpi
    },
    {
        'ensemble': 'nvt',
        'thermostat': 'bussi',
        'device_name': 'cpu',
        'ranks_per_partition': NUM_CPU_RANKS,
        'aggregator': partition_jobs_cpu_mpi
    },
    {
        'ensemble': 'npt',
        'thermostat': 'bussi',
        'device_name': 'cpu',
        'ranks_per_partition': NUM_CPU_RANKS,
        'aggregator': partition_jobs_cpu_mpi
    },
]

if CONFIG["enable_gpu"]:
    md_job_definitions.extend([
        {
            'ensemble': 'nvt',
            'thermostat': 'langevin',
            'device_name': 'gpu',
            'ranks_per_partition': 1,
            'aggregator': partition_jobs_gpu
        },
        {
            'ensemble': 'nvt',
            'thermostat': 'mttk',
            'device_name': 'gpu',
            'ranks_per_partition': 1,
            'aggregator': partition_jobs_gpu
        },
        {
            'ensemble': 'nvt',
            'thermostat': 'bussi',
            'device_name': 'gpu',
            'ranks_per_partition': 1,
            'aggregator': partition_jobs_gpu
        },
        {
            'ensemble': 'npt',
            'thermostat': 'bussi',
            'device_name': 'gpu',
            'ranks_per_partition': 1,
            'aggregator': partition_jobs_gpu
        },
    ])


def add_md_sampling_job(ensemble, thermostat, device_name, ranks_per_partition,
                        aggregator):
    """Add a MD sampling job to the workflow."""
    sim_mode = f'{ensemble}_{thermostat}_md'

    directives = dict(walltime=CONFIG["max_walltime"],
                      executable=CONFIG["executable"],
                      nranks=util.total_ranks_function(ranks_per_partition))

    if device_name == 'gpu':
        directives['ngpu'] = util.total_ranks_function(ranks_per_partition)

    @Project.pre.after(lj_fluid_create_initial_state)
    @Project.post(
        util.gsd_step_greater_equal_function(
            f'{sim_mode}_{device_name}_quantities.gsd', TOTAL_STEPS))
    @Project.operation(name=f'lj_fluid_{sim_mode}_{device_name}',
                       directives=directives,
                       aggregator=aggregator)
    def md_sampling_operation(*jobs):
        """Perform sampling simulation given the definition."""
        import hoomd

        communicator = hoomd.communicator.Communicator(
            ranks_per_partition=ranks_per_partition)
        job = jobs[communicator.partition]

        if communicator.rank == 0:
            print(f'starting lj_fluid_{sim_mode}_{device_name}', job)

        if device_name == 'gpu':
            device_cls = hoomd.device.GPU
        elif device_name == 'cpu':
            device_cls = hoomd.device.CPU

        device = device_cls(
            communicator=communicator,
            message_filename=job.fn(f'run_{sim_mode}_{device_name}.log'))

        run_md_sim(job, device, ensemble, thermostat)

    md_sampling_jobs.append(md_sampling_operation)


for definition in md_job_definitions:
    add_md_sampling_job(**definition)

#################################
# MC simulations
#################################


def make_mc_simulation(job,
                       device,
                       initial_state,
                       sim_mode,
                       extra_loggables=[]):
    """Make an MC Simulation.

    Args:
        job (`signac.job.Job`): Signac job object.
        device (`hoomd.device.Device`): Device object.
        initial_state (str): Path to the gsd file to be used as an initial state
        for the simulation.
        sim_mode (str): String defining the simulation mode.
        extra_loggables (list): List of extra loggables to log to gsd files.
        Patch energies are logged by default.
    """
    import hoomd
    from hoomd import hpmc
    import numpy
    from custom_actions import ComputeDensity

    # integrator
    mc = hpmc.integrate.Sphere(nselect=1)
    mc.shape['A'] = dict(diameter=0.0)

    # pair potential
    epsilon = LJ_PARAMS['epsilon'] / job.sp.kT  # noqa F841
    sigma = LJ_PARAMS['sigma']
    r_on = LJ_PARAMS['r_on']
    r_cut = job.statepoint.r_cut

    # the potential will have xplor smoothing with r_on=2
    lj_str = """// standard lj energy with sigma set to 1
                float rsq = dot(r_ij, r_ij);
                float r_cut = {r_cut};
                float r_cutsq = r_cut * r_cut;

                if (rsq >= r_cutsq)
                    return 0.0f;

                float sigma = {sigma};
                float sigsq = sigma * sigma;
                float rsqinv = sigsq / rsq;
                float r6inv = rsqinv * rsqinv * rsqinv;
                float r12inv = r6inv * r6inv;
                float energy = 4 * {epsilon} * (r12inv - r6inv);

                float r_on = {r_on};
                float r_onsq = r_on * r_on;

                // energy shift for WCA
                if (r_onsq > r_cutsq)
                {{
                    energy += {epsilon};
                }}

                // apply xplor smoothing
                if (rsq > r_onsq)
                {{
                    // computing denominator for the shifting factor
                    float diff = r_cutsq - r_onsq;
                    float denom = diff * diff * diff;

                    // compute second term for the shift
                    float second_term = diff - r_onsq - r_onsq;
                    second_term += (2 * rsq);

                    // compute first term for the shift
                    float first_term = r_cutsq - rsq;
                    first_term = first_term * first_term;
                    float smoothing = first_term * second_term / denom;
                    energy = energy * smoothing;
                }}
                return energy;
            """.format(epsilon=epsilon, sigma=sigma, r_on=r_on, r_cut=r_cut)

    lj_jit_potential = hpmc.pair.user.CPPPotential(r_cut=r_cut,
                                                   code=lj_str,
                                                   param_array=[])
    mc.pair_potential = lj_jit_potential

    # pair force to compute virial pressure
    nlist = hoomd.md.nlist.Cell(buffer=0.4)
    lj = hoomd.md.pair.LJ(default_r_cut=job.statepoint.r_cut,
                    default_r_on=LJ_PARAMS['r_on'],
                    nlist=nlist)
    lj.params[('A', 'A')] = dict(sigma=LJ_PARAMS['sigma'],
                                 epsilon=LJ_PARAMS['epsilon'])
    lj.mode = 'xplor'

    # compute the density
    compute_density = ComputeDensity()

    # log to gsd
    logger_gsd = hoomd.logging.Logger(categories=['scalar', 'sequence'])
    logger_gsd.add(lj_jit_potential, quantities=['energy'])
    logger_gsd.add(mc, quantities=['translate_moves'])
    logger_gsd.add(compute_density)
    for loggable in extra_loggables:
        logger_gsd.add(loggable)

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
    for loggable in extra_loggables:
        # call attach method explicitly so we can access simulation state when
        # computing the loggable quantity
        if hasattr(loggable, 'attach'):
            loggable.attach(sim)

    compute_density.attach(sim)

    def _compute_virial_pressure():
        virials = numpy.sum(lj.virials, 0)
        w = 0
        if virials is not None:
            w = virials[0] + virials[3] + virials[5]
        V = sim.state.box.volume
        return job.statepoint.num_particles * job.statepoint.kT / V + w / (3 * V)

    logger_gsd[('custom', 'virial_pressure')] = (_compute_virial_pressure, 'scalar')

    # move size tuner
    mstuner = hpmc.tune.MoveSize.scale_solver(
        moves=['d'],
        target=0.2,
        max_translation_move=0.5,
        trigger=hoomd.trigger.And([
            hoomd.trigger.Periodic(100),
            hoomd.trigger.Before(RANDOMIZE_STEPS | EQUILIBRATE_STEPS // 2)
        ]))
    sim.operations.add(mstuner)
    sim.operations.computes.append(lj)

    return sim


def run_nvt_mc_sim(job, device):
    """Run MC sim in NVT."""
    import hoomd

    if not hoomd.version.llvm_enabled:
        device.notice("LLVM disabled, skipping MC simulations.")
        return

    # simulation
    initial_state = job.fn('lj_fluid_initial_state.gsd')
    sim_mode = 'nvt_mc'
    sim = make_mc_simulation(job, device, initial_state, sim_mode)

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


def run_npt_mc_sim(job, device):
    """Run MC sim in NPT."""
    import hoomd
    from hoomd import hpmc

    if not hoomd.version.llvm_enabled:
        device.notice("LLVM disabled, skipping MC simulations.")
        return

    # device
    initial_state = job.fn('lj_fluid_initial_state.gsd')
    sim_mode = 'npt_mc'

    # box updates
    boxmc = hpmc.update.BoxMC(betaP=job.statepoint.pressure / job.sp.kT,
                              trigger=hoomd.trigger.Periodic(1))
    boxmc.volume = dict(weight=1.0, mode='ln', delta=0.01)

    # simulation
    sim = make_mc_simulation(job,
                             device,
                             initial_state,
                             sim_mode,
                             extra_loggables=[boxmc])

    sim.operations.add(boxmc)

    boxmc_tuner = hpmc.tune.BoxMCMoveSize.scale_solver(
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
    # equilibration.
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


mc_sampling_jobs = []
mc_job_definitions = [
    {
        'mode': 'nvt',
        'device_name': 'cpu',
        'ranks_per_partition': NUM_CPU_RANKS,
        'aggregator': partition_jobs_cpu_mpi
    },
    {
        'mode': 'npt',
        'device_name': 'cpu',
        'ranks_per_partition': NUM_CPU_RANKS,
        'aggregator': partition_jobs_cpu_mpi
    },
]

if CONFIG["enable_gpu"]:
    mc_job_definitions.extend([
        {
            'mode': 'nvt',
            'device_name': 'gpu',
            'ranks_per_partition': 1,
            'aggregator': partition_jobs_gpu
        },
    ])


def add_mc_sampling_job(mode, device_name, ranks_per_partition, aggregator):
    """Add a MC sampling job to the workflow."""
    directives = dict(walltime=CONFIG["max_walltime"],
                      executable=CONFIG["executable"],
                      nranks=util.total_ranks_function(ranks_per_partition))

    if device_name == 'gpu':
        directives['ngpu'] = util.total_ranks_function(ranks_per_partition)

    @Project.pre.after(lj_fluid_create_initial_state)
    @Project.post(
        util.gsd_step_greater_equal_function(
            f'{mode}_mc_{device_name}_quantities.gsd', TOTAL_STEPS))
    @Project.operation(name=f'lj_fluid_{mode}_mc_{device_name}',
                       directives=directives,
                       aggregator=aggregator)
    def sampling_operation(*jobs):
        """Perform sampling simulation given the definition."""
        import hoomd

        communicator = hoomd.communicator.Communicator(
            ranks_per_partition=ranks_per_partition)
        job = jobs[communicator.partition]

        if communicator.rank == 0:
            print(f'starting lj_fluid_{mode}_mc_{device_name}', job)

        if device_name == 'gpu':
            device_cls = hoomd.device.GPU
        elif device_name == 'cpu':
            device_cls = hoomd.device.CPU

        device = device_cls(
            communicator=communicator,
            message_filename=job.fn(f'run_{mode}_mc_{device_name}.log'))

        globals().get(f'run_{mode}_mc_sim')(job, device)

    mc_sampling_jobs.append(sampling_operation)


if CONFIG['enable_llvm']:
    for definition in mc_job_definitions:
        add_mc_sampling_job(**definition)


@Project.pre(is_lj_fluid)
@Project.pre.after(*md_sampling_jobs)
@Project.pre.after(*mc_sampling_jobs)
# @Project.post.true('lj_fluid_analysis_complete')
@Project.operation(directives=dict(walltime=CONFIG['short_walltime'],
                                   executable=CONFIG["executable"]))
def lj_fluid_analyze(job):
    """Analyze the output of all simulation modes."""
    import gsd.hoomd
    import numpy
    import math
    import matplotlib
    import matplotlib.style
    import matplotlib.figure
    matplotlib.style.use('fivethirtyeight')

    print('starting lj_fluid_analyze:', job)

    sim_modes = [
        'nvt_langevin_md_cpu',
        'nvt_mttk_md_cpu',
        'nvt_bussi_md_cpu',
        'npt_bussi_md_cpu',
    ]

    if os.path.exists(job.fn('nvt_langevin_md_gpu_quantities.gsd')):
        sim_modes.extend([        'nvt_langevin_md_gpu',
                                  'nvt_mttk_md_gpu',
                                'nvt_bussi_md_gpu',
        'npt_bussi_md_gpu',
        ])

    if os.path.exists(job.fn('nvt_mc_cpu_quantities.gsd')):
        sim_modes.extend(['nvt_mc_cpu', 'npt_mc_cpu'])

    if os.path.exists(job.fn('nvt_mc_gpu_quantities.gsd')):
        sim_modes.extend(['nvt_mc_gpu'])

    _sort_sim_modes(sim_modes)

    timesteps = {}
    energies = {}
    pressures = {}
    densities = {}
    linear_momentum = {}
    kinetic_temperature = {}

    for sim_mode in sim_modes:
        log_traj = gsd.hoomd.read_log(job.fn(sim_mode + '_quantities.gsd'))

        timesteps[sim_mode] = log_traj['configuration/step']

        if 'md' in sim_mode:
            energies[sim_mode] = log_traj['log/md/compute/ThermodynamicQuantities/potential_energy']
        else:
            energies[sim_mode] = log_traj['log/hpmc/pair/user/CPPPotential/energy'] * job.statepoint.kT

        energies[sim_mode] /=  job.statepoint.num_particles

        if 'md' in sim_mode:
            pressures[sim_mode] = log_traj['log/md/compute/ThermodynamicQuantities/pressure']
        else:
            pressures[sim_mode] = log_traj['log/custom/virial_pressure']

        densities[sim_mode] = log_traj['log/custom_actions/ComputeDensity/density']

        if 'md' in sim_mode and 'langevin' not in sim_mode:
            momentum_vector = log_traj['log/md/Integrator/linear_momentum']
            linear_momentum[sim_mode] = [
                math.sqrt(v[0]**2 + v[1]**2 + v[2]**2) for v in momentum_vector
            ]
        else:
            linear_momentum[sim_mode] = numpy.zeros(len(energies[sim_mode]))

    # save averages
    for mode in sim_modes:
        job.document[mode] = dict(pressure=float(numpy.mean(pressures[mode])),
                                  potential_energy=float(
                                      numpy.mean(energies[mode])),
                                  density=float(numpy.mean(densities[mode])))

    # Plot results
    fig = matplotlib.figure.Figure(figsize=(20, 20 / 3.24 * 2), layout='tight')
    ax = fig.add_subplot(2, 2, 1)
    util.plot_timeseries(ax=ax,
         timesteps=timesteps,
         data=densities,
         ylabel=r"$\rho$",
         expected=job.sp.density, max_points=500)
    ax.legend()

    ax = fig.add_subplot(2, 2, 2)
    util.plot_timeseries(ax=ax, timesteps=timesteps, data=pressures, ylabel=r"$P$", expected=job.sp.pressure, max_points=500)

    ax = fig.add_subplot(2, 2, 3)
    util.plot_timeseries(ax=ax, timesteps=timesteps, data=energies, ylabel="$U / N$", max_points=500)

    ax = fig.add_subplot(2, 2, 4)
    util.plot_timeseries(ax=ax,
         timesteps=timesteps,
         data={
             mode: numpy.asarray(lm) / job.sp.num_particles
             for mode, lm in linear_momentum.items()
         },
         ylabel=r'$|\vec{p}| / N$', max_points=500)

    fig.suptitle(f"$kT={job.statepoint.kT}$, $\\rho={job.statepoint.density}$, "
                 f"$N={job.statepoint.num_particles}$, "
                 f"$r_\\mathrm{{cut}}={job.statepoint.r_cut}$, "
                 f"replicate={job.statepoint.replicate_idx}")
    fig.savefig(job.fn('nvt_npt_plots.svg'), bbox_inches='tight')

    job.document['lj_fluid_analysis_complete'] = True


analysis_aggregator = aggregator.groupby(key=['kT', 'density', 'num_particles', 'r_cut'],
                                         sort_by='replicate_idx',
                                         select=is_lj_fluid)


@Project.pre(
    lambda *jobs: util.true_all(*jobs, key='lj_fluid_analysis_complete'))
@Project.post(
    lambda *jobs: util.true_all(*jobs, key='lj_fluid_compare_modes_complete'))
@Project.operation(directives=dict(walltime=CONFIG['short_walltime'],
                                   executable=CONFIG["executable"]),
                   aggregator=analysis_aggregator)
def lj_fluid_compare_modes(*jobs):
    """Compares the tested simulation modes."""
    import numpy
    import matplotlib
    import matplotlib.style
    import matplotlib.figure
    import scipy.stats
    matplotlib.style.use('fivethirtyeight')

    print('starting lj_fluid_compare_modes:', jobs[0])

    sim_modes = [
        'nvt_langevin_md_cpu',
        'nvt_mttk_md_cpu',
        'nvt_bussi_md_cpu',
        'npt_bussi_md_cpu',
    ]

    if os.path.exists(jobs[0].fn('nvt_langevin_md_gpu_quantities.gsd')):
        sim_modes.extend(['nvt_langevin_md_gpu', 'nvt_mttk_md_gpu', 'nvt_bussi_md_gpu', 'npt_bussi_md_gpu',])

    if os.path.exists(jobs[0].fn('nvt_mc_cpu_quantities.gsd')):
        sim_modes.extend(['nvt_mc_cpu', 'npt_mc_cpu'])

    if os.path.exists(jobs[0].fn('nvt_mc_gpu_quantities.gsd')):
        sim_modes.extend(['nvt_mc_gpu'])

    _sort_sim_modes(sim_modes)

    quantity_names = ['density', 'pressure', 'potential_energy']
    labels = {'density': r'$\frac{\rho_\mathrm{sample} - \rho}{\rho} \cdot 1000$',
              'pressure': r'$\frac{P_\mathrm{sample} - P}{P} \cdot 1000$',
              'potential_energy': r'$\frac{U_\mathrm{sample} - <U>}{<U>} \cdot 1000$',
        }

    # grab the common statepoint parameters
    kT = jobs[0].sp.kT
    set_density = jobs[0].sp.density
    set_pressure = jobs[0].sp.pressure
    num_particles = jobs[0].sp.num_particles

    quantity_reference = dict(density=set_density,
                              pressure=set_pressure,
                              potential_energy=None)

    fig = matplotlib.figure.Figure(figsize=(8, 8 / 1.618 * 3), layout='tight')
    fig.suptitle(f"$kT={kT}$, $\\rho={set_density}$, $r_\\mathrm{{cut}}={jobs[0].statepoint.r_cut}$, $N={num_particles}$")

    for i, quantity_name in enumerate(quantity_names):
        ax = fig.add_subplot(3, 1, i + 1)

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

        # compute the energy differences
        quantity_list = [avg_quantity[mode] for mode in sim_modes]
        stderr_list = numpy.array([stderr_quantity[mode] for mode in sim_modes])

        if quantity_reference[quantity_name] is not None:
            reference = quantity_reference[quantity_name]
        else:
            reference = numpy.mean(quantity_list)

        quantity_diff_list = numpy.array(quantity_list) - reference

        ax.errorbar(x=range(len(sim_modes)),
                    y=quantity_diff_list / reference * 1000,
                    yerr=numpy.fabs(stderr_list / reference * 1000),
                    fmt='s')
        ax.set_xticks(range(len(sim_modes)), sim_modes, rotation=45)
        ax.set_ylabel(labels[quantity_name])

        # Indicate average nvt and npt values separately.
        npt_modes = list(filter(lambda x: 'npt' in x, sim_modes))
        npt_mean = numpy.mean([avg_quantity[mode] for mode in npt_modes])
        nvt_modes = list(filter(lambda x: 'nvt' in x, sim_modes))
        nvt_mean = numpy.mean([avg_quantity[mode] for mode in nvt_modes])

        # _sort_sim_modes places npt modes first
        ax.hlines(y=(npt_mean - reference) / reference * 1000,
                  xmin=0,
                  xmax=len(npt_modes) - 1,
                  linestyles='dashed',
                  colors='k')

        ax.hlines(y=(nvt_mean - reference) / reference * 1000,
                  xmin=len(npt_modes),
                  xmax=len(sim_modes) - 1,
                  linestyles='dashed',
                  colors='k')

        if quantity_name == "density":
            print(f"Average npt_mc_cpu density {num_particles}:", avg_quantity['npt_mc_cpu'], '+/-', stderr_quantity['npt_mc_cpu'])
            print(f"Average npt_md_cpu density {num_particles}:", avg_quantity['npt_bussi_md_cpu'], '+/-', stderr_quantity['npt_bussi_md_cpu'])
        if quantity_name == "pressure":
            print(f"Average nvt_mc_cpu pressure {num_particles}:", avg_quantity['nvt_mc_cpu'], '+/-', stderr_quantity['nvt_mc_cpu'])
        if quantity_name == "pressure":
            print(f"Average npt_mc_cpu pressure {num_particles}:", avg_quantity['npt_mc_cpu'], '+/-', stderr_quantity['npt_mc_cpu'])


    filename = f'lj_fluid_compare_kT{kT}_density{round(set_density, 2)}_' \
               f'r_cut{round(jobs[0].statepoint.r_cut, 2)}_' \
               f'N{num_particles}.svg'

    fig.savefig(os.path.join(jobs[0]._project.path, filename),
                bbox_inches='tight')

    for job in jobs:
        job.document['lj_fluid_compare_modes_complete'] = True


@Project.pre.after(*md_sampling_jobs)
@Project.post(
    lambda *jobs: util.true_all(*jobs, key='lj_fluid_distribution_analyze_complete'))
@Project.operation(directives=dict(walltime=CONFIG['short_walltime'],
                                   executable=CONFIG["executable"]),
                   aggregator=analysis_aggregator)
def lj_fluid_distribution_analyze(*jobs):
    """Checks that MD follows the correct KE distribution."""
    import gsd.hoomd
    import numpy
    import matplotlib
    import matplotlib.style
    import matplotlib.figure
    import scipy
    matplotlib.style.use('fivethirtyeight')

    print('starting lj_fluid_distribution_analyze:', jobs[0])

    sim_modes = [
        'nvt_langevin_md_cpu',
        'nvt_mttk_md_cpu',
        'nvt_bussi_md_cpu',
        'npt_bussi_md_cpu',
    ]

    if os.path.exists(jobs[0].fn('nvt_langevin_md_gpu_quantities.gsd')):
        sim_modes.extend([        'nvt_langevin_md_gpu',
                                  'nvt_mttk_md_gpu',
                                'nvt_bussi_md_gpu',
        'npt_bussi_md_gpu',
        ])

    if os.path.exists(jobs[0].fn('nvt_mc_cpu_quantities.gsd')):
        sim_modes.extend(['nvt_mc_cpu', 'npt_mc_cpu'])

    if os.path.exists(jobs[0].fn('nvt_mc_gpu_quantities.gsd')):
        sim_modes.extend(['nvt_mc_gpu'])

    _sort_sim_modes(sim_modes)

    # grab the common statepoint parameters
    kT = jobs[0].sp.kT
    set_density = jobs[0].sp.density
    num_particles = jobs[0].sp.num_particles

    fig = matplotlib.figure.Figure(figsize=(20, 20 / 3.24 * 2), layout='tight')
    fig.suptitle(f"$kT={kT}$, $\\rho={set_density}$, $r_\\mathrm{{cut}}={jobs[0].statepoint.r_cut}$, $N={num_particles}$")

    ke_means_expected = collections.defaultdict(list)
    ke_sigmas_expected = collections.defaultdict(list)
    ke_samples = collections.defaultdict(list)
    potential_energy_samples = collections.defaultdict(list)
    density_samples = collections.defaultdict(list)
    pressure_samples = collections.defaultdict(list)

    for job in jobs:
        for sim_mode in sim_modes:

            if sim_mode.startswith('nvt_langevin'):
                n_dof = num_particles * 3
            else:
                n_dof = num_particles * 3 - 3

            print('Reading' + job.fn(sim_mode + '_quantities.gsd'))
            log_traj = gsd.hoomd.read_log(job.fn(sim_mode + '_quantities.gsd'))

            if 'md' in sim_mode:
                ke = log_traj['log/md/compute/ThermodynamicQuantities/kinetic_energy']
                ke_means_expected[sim_mode].append(numpy.mean(ke) - 1 / 2 * n_dof * kT)
                ke_sigmas_expected[sim_mode].append(numpy.std(ke) - 1 / math.sqrt(2) * math.sqrt(n_dof) * kT)

                ke_samples[sim_mode].extend(list(ke))
            else:
                ke_samples[sim_mode].extend([3/2 * job.statepoint.num_particles * job.statepoint.kT])

            if 'md' in sim_mode:
                potential_energy_samples[sim_mode].extend(list(log_traj['log/md/compute/ThermodynamicQuantities/potential_energy']))
            else:
                potential_energy_samples[sim_mode].extend(list(log_traj['log/hpmc/pair/user/CPPPotential/energy'] * job.statepoint.kT))

            if 'md' in sim_mode:
                pressure_samples[sim_mode].extend(list(log_traj['log/md/compute/ThermodynamicQuantities/pressure']))
            else:
                pressure_samples[sim_mode].extend(list(log_traj['log/custom/virial_pressure']))

            if 'nvt' in sim_mode:
                density_samples[sim_mode].extend(list(log_traj['log/custom_actions/ComputeDensity/density']))
            else:
                density_samples[sim_mode].extend(list(log_traj['log/custom_actions/ComputeDensity/density']))

    ax = fig.add_subplot(2, 2, 1)
    util.plot_vs_expected(ax, ke_means_expected, '$<K> - 1/2 N_{dof} k T$')

    ax = fig.add_subplot(2, 2, 2)
    # https://doi.org/10.1371/journal.pone.0202764
    util.plot_vs_expected(ax, ke_sigmas_expected, r'$\Delta K - 1/\sqrt{2} \sqrt{N_{dof}} k T$')

    ax = fig.add_subplot(2,4,5)
    rv = scipy.stats.gamma(3 * job.statepoint.num_particles/2, scale=job.statepoint.kT)
    util.plot_distribution(ax, ke_samples, 'K', expected=rv.pdf)
    ax.legend(loc='upper right', fontsize='xx-small')

    ax = fig.add_subplot(2,4,6)
    util.plot_distribution(ax, potential_energy_samples, 'U')

    ax = fig.add_subplot(2,4,7)
    util.plot_distribution(ax, density_samples, r'$\rho$', expected=job.statepoint.density)

    ax = fig.add_subplot(2,4,8)
    util.plot_distribution(ax, pressure_samples, 'P', expected=job.statepoint.pressure)

    filename = f'lj_fluid_distribution_analyze_kT{kT}'\
               f'_density{round(set_density, 2)}_' \
               f'r_cut{round(jobs[0].statepoint.r_cut, 2)}_' \
               f'N{num_particles}.svg'
    fig.savefig(os.path.join(jobs[0]._project.path, filename),
                bbox_inches='tight')

    for job in jobs:
        job.document['lj_fluid_distribution_analyze_complete'] = True


#################################
# MD conservation simulations
#################################


def run_nve_md_sim(job, device, run_length):
    """Run the MD simulation in NVE."""
    import hoomd

    sim_mode = 'nve_md'
    restart_filename = util.get_job_filename(sim_mode, device, 'restart', 'gsd')
    is_restarting = job.isfile(restart_filename)

    if is_restarting:
        initial_state = job.fn(restart_filename)
    else:
        initial_state = job.fn('lj_fluid_initial_state.gsd')

    nve = hoomd.md.methods.ConstantVolume(hoomd.filter.All())

    sim = make_md_simulation(job,
                             device,
                             initial_state,
                             nve,
                             sim_mode,
                             period_multiplier=400)

    if not is_restarting:
        sim.state.thermalize_particle_momenta(hoomd.filter.All(), job.sp.kT)

    # Run for a long time to look for energy and momentum drift
    device.notice('Running...')

    util.run_up_to_walltime(
        sim=sim,
        end_step=RANDOMIZE_STEPS + EQUILIBRATE_STEPS + run_length,
        steps=500_000,
        walltime_stop=CONFIG["max_walltime"] * 3600 - 10 * 60)

    if sim.timestep == RANDOMIZE_STEPS + EQUILIBRATE_STEPS + run_length:
        device.notice('Done.')
    else:
        device.notice('Ending run early due to walltime limits at:'
                      f'{device.communicator.walltime}')

    hoomd.write.GSD.write(state=sim.state,
                          filename=job.fn(restart_filename),
                          mode='wb')


def is_lj_fluid_nve(job):
    """Test if a given job should be run for NVE conservation."""
    return job.statepoint['subproject'] == 'lj_fluid' and \
        job.statepoint['replicate_idx'] < NUM_NVE_RUNS


partition_jobs_cpu_mpi_nve = aggregator.groupsof(num=min(
    CONFIG["replicates"], CONFIG["max_cores_submission"] // NUM_CPU_RANKS),
                                                 sort_by=sort_key,
                                                 select=is_lj_fluid_nve)

partition_jobs_gpu_nve = aggregator.groupsof(num=min(
    CONFIG["replicates"], CONFIG["max_gpus_submission"]),
                                             sort_by=sort_key,
                                             select=is_lj_fluid_nve)


nve_md_sampling_jobs = []
nve_md_job_definitions = [
    {
        'device_name': 'cpu',
        'ranks_per_partition': NUM_CPU_RANKS,
        'aggregator': partition_jobs_cpu_mpi_nve,
        'run_length': 200_000_000,
    },
]

if CONFIG["enable_gpu"]:
    nve_md_job_definitions.extend([
        {
            'device_name': 'gpu',
            'ranks_per_partition': 1,
            'aggregator': partition_jobs_gpu_nve,
            'run_length': 800_000_000,
        },
    ])


def add_nve_md_job(device_name, ranks_per_partition, aggregator, run_length):
    """Add a MD NVE conservation job to the workflow."""
    sim_mode = 'nve_md'

    directives = dict(walltime=CONFIG["max_walltime"],
                      executable=CONFIG["executable"],
                      nranks=util.total_ranks_function(ranks_per_partition))

    if device_name == 'gpu':
        directives['ngpu'] = util.total_ranks_function(ranks_per_partition)

    @Project.pre.after(lj_fluid_create_initial_state)
    @Project.post(
        util.gsd_step_greater_equal_function(
            f'{sim_mode}_{device_name}_quantities.gsd', run_length))
    @Project.operation(name=f'lj_fluid_{sim_mode}_{device_name}',
                       directives=directives,
                       aggregator=aggregator)
    def lj_fluid_nve_md_job(*jobs):
        """Run NVE MD."""
        import hoomd

        communicator = hoomd.communicator.Communicator(
            ranks_per_partition=ranks_per_partition)
        job = jobs[communicator.partition]

        if communicator.rank == 0:
            print(f'starting lj_fluid_{sim_mode}_{device_name}:', job)

        if device_name == 'gpu':
            device_cls = hoomd.device.GPU
        elif device_name == 'cpu':
            device_cls = hoomd.device.CPU

        device = device_cls(communicator=communicator,
                            message_filename=job.fn(f'run_{sim_mode}_{device_name}.log'))
        run_nve_md_sim(job, device, run_length=run_length)

    nve_md_sampling_jobs.append(lj_fluid_nve_md_job)


for definition in nve_md_job_definitions:
    add_nve_md_job(**definition)


@Project.pre.after(*nve_md_sampling_jobs)
@Project.post(lambda *jobs: util.true_all(
    *jobs[0:NUM_NVE_RUNS], key='lj_fluid_conservation_analysis_complete'))
@Project.operation(directives=dict(walltime=CONFIG['short_walltime'],
                                   executable=CONFIG["executable"]),
                   aggregator=analysis_aggregator)
def lj_fluid_conservation_analyze(*jobs):
    """Analyze the output of NVE simulations and inspect conservation."""
    import gsd.hoomd
    import numpy
    import math
    import matplotlib
    import matplotlib.style
    import matplotlib.figure
    matplotlib.style.use('fivethirtyeight')

    print('starting lj_fluid_conservation_analyze:', jobs[0])

    sim_modes = ['nve_md_cpu']
    if os.path.exists(jobs[0].fn('nve_md_gpu_quantities.gsd')):
        sim_modes.extend['nve_md_gpu']

    jobs = jobs[0:NUM_NVE_RUNS]

    energies = []
    linear_momenta = []

    for job in jobs:
        job_energies = {}
        job_linear_momentum = {}

        for sim_mode in sim_modes:
            log_traj = gsd.hoomd.read_log(job.fn(sim_mode + '_quantities.gsd'))

            job_energies[sim_mode] = (
                log_traj['log/md/compute/ThermodynamicQuantities/potential_energy']
                + log_traj['log/md/compute/ThermodynamicQuantities/kinetic_energy'])
            job_energies[sim_mode] = (
                job_energies[sim_mode]
                - job_energies[sim_mode][0]) / job.statepoint["num_particles"]

            momentum_vector = log_traj['log/md/Integrator/linear_momentum']
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

    fig.suptitle("LJ conservation tests: "
                 f"$kT={job.statepoint.kT}$, $\\rho={job.statepoint.density}$, "
                 f"$r_\\mathrm{{cut}}={job.statepoint.r_cut}$, "
                 f"$N={job.statepoint.num_particles}$")
    filename = f'lj_fluid_conservation_kT{job.statepoint.kT}_' \
               f'density{round(job.statepoint.density, 2)}_' \
               f'r_cut{round(jobs[0].statepoint.r_cut, 2)}_' \
               f'N{job.statepoint.num_particles}.svg'

    fig.savefig(os.path.join(jobs[0]._project.path, filename),
                bbox_inches='tight')

    for job in jobs:
        job.document['lj_fluid_conservation_analysis_complete'] = True
