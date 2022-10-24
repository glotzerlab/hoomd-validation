# Copyright (c) 2022 The Regents of the University of Michigan.
# Part of HOOMD-blue, released under the BSD 3-Clause License.

"""Lennard Jones phase behavior validation test."""

from config import test_project_dict, CONFIG
from project_classes import LJFluid
from flow import aggregator
import os

# Run parameters shared between simulations
RANDOMIZE_STEPS = 5e4
RUN_STEPS = 5e5
WRITE_PERIOD = 4000
LOG_PERIOD = {'trajectory': 50000, 'quantities': 2000}
FRAMES_ANALYZE = int(RUN_STEPS / LOG_PERIOD['quantities'] * 1 / 2)
LJ_PARAMS = {'epsilon': 1.0, 'sigma': 1.0, 'r_on': 2.0, 'r_cut': 2.5}


@LJFluid.operation(directives=dict(executable=CONFIG["executable"], nranks=8))
@LJFluid.post.isfile('initial_state.gsd')
def create_initial_state(job):
    """Create initial system configuration."""
    import hoomd
    import numpy
    import itertools

    sp = job.sp
    device = hoomd.device.CPU(msg_file=job.fn('create_initial_state.log'))

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

    sim = hoomd.Simulation(device=device, seed=job.statepoint.replicate_idx)
    sim.create_state_from_snapshot(snap)
    sim.operations.integrator = mc

    device.notice('Randomizing initial state...')
    sim.run(RANDOMIZE_STEPS)
    device.notice(f'Done. Move counts: {mc.translate_moves}')

    hoomd.write.GSD.write(state=sim.state,
                          filename=job.fn("initial_state.gsd"),
                          mode='wb')


def make_simulation(job, device, initial_state, integrator, sim_mode, logger):
    """Make a simulation.

    This operation returns a simulation with only the things needed for each
    type of simulation. This includes initial state, random seed, integrator,
    table writer, and trajectory writer.

    Args:
        job (`signac.Job`): signac job object.
        device (`hoomd.device.Device`): hoomd device object.
        initial_state (str): Path to the gsd file to be used as an initial
        state.
        integrator (`hoomd.md.Integrator`): hoomd integrator object.
        sim_mode (str): String defining the simulation mode.
        logger (`hoomd.logging.Logger`): Logger object. All logged quantities
        should be added before being passed to this function
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


def make_md_simulation(job,
                       device,
                       initial_state,
                       method,
                       sim_mode,
                       extra_loggables=[]):
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
    """
    import hoomd
    from hoomd import md

    # pair force
    nlist = md.nlist.Cell(buffer=0.4)
    lj = md.pair.LJ(default_r_cut=LJ_PARAMS['r_cut'],
                    default_r_on=LJ_PARAMS['r_on'],
                    nlist=nlist)
    lj.params[('A', 'A')] = dict(sigma=LJ_PARAMS['sigma'],
                                 epsilon=LJ_PARAMS['epsilon'])
    lj.mode = 'xplor'

    # integrator
    integrator = md.Integrator(dt=0.001953125, methods=[method], forces=[lj])

    # compute thermo
    thermo = md.compute.ThermodynamicQuantities(hoomd.filter.All())

    # add gsd log quantities
    logger = hoomd.logging.Logger(categories=['scalar', 'sequence'])
    logger.add(
        thermo,
        quantities=['pressure', 'potential_energy', 'kinetic_temperature'])
    logger.add(integrator, quantities=['linear_momentum'])
    for loggable in extra_loggables:
        logger.add(loggable)

    # simulation
    sim = make_simulation(job, device, initial_state, integrator, sim_mode,
                          logger)
    sim.operations.add(thermo)
    for loggable in extra_loggables:
        # call attach explicitly so we can access sim state when computing the
        # loggable quantity
        if hasattr(loggable, 'attach'):
            loggable.attach(sim)

    # thermalize momenta
    sim.state.thermalize_particle_momenta(hoomd.filter.All(), job.sp.kT)

    return sim


def run_langevin_md_sim(job, device):
    """Run the MD simulation in Langevin."""
    import hoomd
    from hoomd import md

    initial_state = job.fn('initial_state.gsd')
    langevin = md.methods.Langevin(hoomd.filter.All(), kT=job.sp.kT)
    langevin.gamma.default = 1.0

    sim_mode = 'langevin_md'

    sim = make_md_simulation(job, device, initial_state, langevin, sim_mode)

    # equilibrate
    device.notice('Equilibrating...')
    sim.run(RANDOMIZE_STEPS)
    device.notice('Done.')

    # run
    device.notice('Running...')
    sim.run(RUN_STEPS)
    device.notice('Done.')


@LJFluid.operation(directives=dict(walltime=2,
                                   executable=CONFIG["executable"],
                                   nranks=8))
@LJFluid.pre.after(create_initial_state)
@LJFluid.post.true('langevin_md_cpu_complete')
def langevin_md_cpu(job):
    """Run Langevin MD on the CPU."""
    import hoomd
    device = hoomd.device.CPU(msg_file=job.fn('run_langevin_md_cpu.log'))
    run_langevin_md_sim(job, device)

    if device.communicator.rank == 0:
        job.document['langevin_md_cpu_complete'] = True


@LJFluid.operation(directives=dict(walltime=2,
                                   executable=CONFIG["executable"],
                                   nranks=1,
                                   ngpu=1))
@LJFluid.pre.after(create_initial_state)
@LJFluid.post.true('langevin_md_gpu_complete')
def langevin_md_gpu(job):
    """Run Langevin MD on the GPU."""
    import hoomd
    device = hoomd.device.GPU(msg_file=job.fn('run_langevin_md_gpu.log'))
    run_langevin_md_sim(job, device)

    if device.communicator.rank == 0:
        job.document['langevin_md_gpu_complete'] = True


def run_nvt_md_sim(job, device):
    """Run the MD simulation in NVT."""
    import hoomd
    from hoomd import md

    initial_state = job.fn('initial_state.gsd')
    nvt = md.methods.NVT(hoomd.filter.All(), kT=job.sp.kT, tau=0.25)
    sim_mode = 'nvt_md'

    sim = make_md_simulation(job, device, initial_state, nvt, sim_mode)

    # thermalize the thermostat
    sim.run(0)
    nvt.thermalize_thermostat_dof()

    # equilibrate
    device.notice('Equilibrating...')
    sim.run(RANDOMIZE_STEPS)
    device.notice('Done.')

    # run
    device.notice('Running...')
    sim.run(RUN_STEPS)
    device.notice('Done.')


@LJFluid.operation(directives=dict(walltime=2,
                                   executable=CONFIG["executable"],
                                   nranks=8))
@LJFluid.pre.after(create_initial_state)
@LJFluid.post.true('nvt_md_cpu_complete')
def nvt_md_cpu(job):
    """Run NVT MD on the CPU."""
    import hoomd
    device = hoomd.device.CPU(msg_file=job.fn('run_nvt_md_cpu.log'))
    run_nvt_md_sim(job, device)

    if device.communicator.rank == 0:
        job.document['nvt_md_cpu_complete'] = True


@LJFluid.operation(directives=dict(walltime=2,
                                   executable=CONFIG["executable"],
                                   nranks=1,
                                   ngpu=1))
@LJFluid.pre.after(create_initial_state)
@LJFluid.post.true('nvt_md_gpu_complete')
def nvt_md_gpu(job):
    """Run NVT MD on the GPU."""
    import hoomd
    device = hoomd.device.GPU(msg_file=job.fn('run_nvt_md_gpu.log'))
    run_nvt_md_sim(job, device)

    if device.communicator.rank == 0:
        job.document['nvt_md_gpu_complete'] = True


def run_npt_md_sim(job, device):
    """Run an NPT simulation at the pressure computed by the NVT simulation."""
    import hoomd
    from hoomd import md
    from custom_actions import ComputeDensity

    initial_state = job.fn('initial_state.gsd')
    p = job.statepoint.pressure
    nvt = md.methods.NVT(hoomd.filter.All(), kT=job.sp.kT, tau=0.25)
    npt = md.methods.NPT(hoomd.filter.All(),
                         kT=job.sp.kT,
                         tau=0.25,
                         S=[p, p, p, 0, 0, 0],
                         tauS=3,
                         couple='xyz')
    sim_mode = 'npt_md'
    density_compute = ComputeDensity()

    sim = make_md_simulation(job,
                             device,
                             initial_state,
                             nvt,
                             sim_mode,
                             extra_loggables=[density_compute])

    # thermalize the thermostat
    sim.run(0)
    nvt.thermalize_thermostat_dof()

    # equilibrate in NVT
    device.notice('Equilibrating...')
    sim.run(RANDOMIZE_STEPS)
    device.notice('Done.')

    # switch to NPT
    # thermalize the thermostat and barostat
    sim.operations.integrator.methods[0] = npt

    # run
    device.notice('Running...')
    sim.run(RUN_STEPS)
    device.notice('Done.')


@LJFluid.operation(directives=dict(walltime=2,
                                   executable=CONFIG["executable"],
                                   nranks=8))
@LJFluid.pre.after(create_initial_state)
@LJFluid.post.true('npt_md_cpu_complete')
def npt_md_cpu(job):
    """Run NPT MD on the CPU."""
    import hoomd
    device = hoomd.device.CPU(msg_file=job.fn('run_npt_md_cpu.log'))
    run_npt_md_sim(job, device)

    if device.communicator.rank == 0:
        job.document['npt_md_cpu_complete'] = True


@LJFluid.operation(directives=dict(walltime=2,
                                   executable=CONFIG["executable"],
                                   nranks=1,
                                   ngpu=1))
@LJFluid.pre.after(create_initial_state)
@LJFluid.post.true('npt_md_gpu_complete')
def npt_md_gpu(job):
    """Run NPT MD on the GPU."""
    import hoomd
    device = hoomd.device.GPU(msg_file=job.fn('run_npt_md_gpu.log'))
    run_npt_md_sim(job, device)

    if device.communicator.rank == 0:
        job.document['npt_md_gpu_complete'] = True


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

    # integrator
    mc = hpmc.integrate.Sphere(nselect=1)
    mc.shape['A'] = dict(diameter=0.0)

    # pair potential
    epsilon = LJ_PARAMS['epsilon'] / job.sp.kT  # noqa F841
    sigma = LJ_PARAMS['sigma']
    r_on = LJ_PARAMS['r_on']
    r_cut = LJ_PARAMS['r_cut']

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

                // apply xplor smoothing
                float r_on = {r_on};
                float r_onsq = r_on * r_on;
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

    # log to gsd
    logger_gsd = hoomd.logging.Logger(categories=['scalar', 'sequence'])
    logger_gsd.add(lj_jit_potential, quantities=['energy'])
    logger_gsd.add(mc, quantities=['translate_moves'])
    for loggable in extra_loggables:
        logger_gsd.add(loggable)

    # make simulation
    sim = make_simulation(job, device, initial_state, mc, sim_mode, logger_gsd)
    for loggable in extra_loggables:
        # call attach method explicitly so we can access simulation state when
        # computing the loggable quantity
        if hasattr(loggable, 'attach'):
            loggable.attach(sim)

    # move size tuner
    mstuner = hpmc.tune.MoveSize.scale_solver(
        moves=['d'],
        target=0.2,
        max_translation_move=0.5,
        trigger=hoomd.trigger.And([
            hoomd.trigger.Periodic(100),
            hoomd.trigger.Before(sim.timestep + int(RANDOMIZE_STEPS))
        ]))
    sim.operations.add(mstuner)

    return sim


def run_nvt_mc_sim(job, device):
    """Run MC sim in NVT."""
    # simulation
    initial_state = job.fn('initial_state.gsd')
    sim_mode = 'nvt_mc'
    sim = make_mc_simulation(job, device, initial_state, sim_mode)

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


@LJFluid.operation(directives=dict(walltime=2,
                                   executable=CONFIG["executable"],
                                   nranks=8))
@LJFluid.pre.after(create_initial_state)
@LJFluid.post.true('nvt_mc_cpu_complete')
def nvt_mc_cpu(job):
    """Run NVT MC on the CPU."""
    import hoomd
    device = hoomd.device.CPU(msg_file=job.fn('run_nvt_mc_cpu.log'))
    run_nvt_mc_sim(job, device)

    if device.communicator.rank == 0:
        job.document['nvt_mc_cpu_complete'] = True


@LJFluid.operation(directives=dict(walltime=2,
                                   executable=CONFIG["executable"],
                                   nranks=1,
                                   ngpu=1))
@LJFluid.pre.after(create_initial_state)
@LJFluid.post.true('nvt_mc_gpu_complete')
def nvt_mc_gpu(job):
    """Run NVT MC on the GPU."""
    import hoomd
    device = hoomd.device.GPU(msg_file=job.fn('run_nvt_mc_gpu.log'))
    run_nvt_mc_sim(job, device)

    if device.communicator.rank == 0:
        job.document['nvt_mc_gpu_complete'] = True


def run_npt_mc_sim(job, device):
    """Run MC sim in NPT."""
    import hoomd
    from hoomd import hpmc
    from custom_actions import ComputeDensity

    # device
    initial_state = job.fn('initial_state.gsd')
    sim_mode = 'npt_mc'

    # compute the density
    compute_density = ComputeDensity()

    # box updates
    boxmc = hpmc.update.BoxMC(betaP=job.statepoint.pressure / job.sp.kT,
                              trigger=hoomd.trigger.Periodic(1))
    boxmc.volume = dict(weight=1.0, mode='ln', delta=0.001)

    # simulation
    sim = make_mc_simulation(job,
                             device,
                             initial_state,
                             sim_mode,
                             extra_loggables=[compute_density, boxmc])

    sim.operations.add(boxmc)

    boxmc_tuner = hpmc.tune.BoxMCMoveSize.scale_solver(
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


@LJFluid.operation(directives=dict(walltime=2,
                                   executable=CONFIG["executable"],
                                   nranks=8))
@LJFluid.pre.after(create_initial_state)
@LJFluid.post.true('npt_mc_cpu_complete')
def npt_mc_cpu(job):
    """Run NPT MC on the CPU."""
    import hoomd
    device = hoomd.device.CPU(msg_file=job.fn('run_npt_mc_cpu.log'))
    run_npt_mc_sim(job, device)

    if device.communicator.rank == 0:
        job.document['npt_mc_cpu_complete'] = True


@LJFluid.operation(directives=dict(walltime=1, executable=CONFIG["executable"]))
@LJFluid.pre.after(langevin_md_cpu)
@LJFluid.pre.after(langevin_md_gpu)
@LJFluid.pre.after(nvt_md_cpu)
@LJFluid.pre.after(nvt_md_gpu)
@LJFluid.pre.after(npt_md_cpu)
@LJFluid.pre.after(npt_md_gpu)
@LJFluid.pre.after(nvt_mc_cpu)
@LJFluid.pre.after(nvt_mc_gpu)
@LJFluid.pre.after(npt_mc_cpu)
@LJFluid.post.true('analysis_complete')
def analyze(job):
    """Analyze the output of all simulation modes."""
    import gsd.hoomd
    import numpy
    import math
    import matplotlib
    import matplotlib.style
    import matplotlib.figure
    matplotlib.style.use('ggplot')
    from util import read_gsd_log_trajectory, get_log_quantity

    constant = dict(
        langevin_md_cpu='density',
        langevin_md_gpu='density',
        nvt_md_cpu='density',
        nvt_md_gpu='density',
        nvt_mc_cpu='density',
        nvt_mc_gpu='density',
        npt_md_cpu='pressure',
        npt_md_gpu='pressure',
        npt_mc_cpu='pressure',
        npt_mc_gpu='pressure',
    )
    sim_modes = [
        'langevin_md_cpu', 'langevin_md_gpu', 'nvt_md_cpu', 'nvt_md_gpu',
        'npt_md_cpu', 'npt_md_gpu', 'nvt_mc_cpu', 'nvt_mc_gpu', 'npt_mc_cpu'
    ]

    energies = {}
    pressures = {}
    densities = {}
    linear_momentum = {}
    kinetic_temperature = {}

    for sim_mode in sim_modes:
        with gsd.hoomd.open(job.fn(sim_mode + '_quantities.gsd')) as gsd_traj:
            # read GSD file once
            traj = read_gsd_log_trajectory(gsd_traj)

        if 'md' in sim_mode:
            energies[sim_mode] = get_log_quantity(
                traj, 'md/compute/ThermodynamicQuantities/potential_energy')
        else:
            energies[sim_mode] = numpy.array(
                get_log_quantity(
                    traj,
                    'hpmc/pair/user/CPPPotential/energy')) * job.statepoint.kT

        if constant[sim_mode] == 'density' and 'md' in sim_mode:
            pressures[sim_mode] = get_log_quantity(
                traj, 'md/compute/ThermodynamicQuantities/pressure')
        elif constant[sim_mode] == 'density' and 'mc' in sim_mode:
            pressures[sim_mode] = numpy.full(len(energies[sim_mode]), numpy.nan)
        else:
            pressures[sim_mode] = numpy.ones(len(
                energies[sim_mode])) * job.statepoint.pressure

        if constant[sim_mode] == 'pressure':
            densities[sim_mode] = get_log_quantity(
                traj, 'custom_actions/ComputeDensity/density')
        else:
            densities[sim_mode] = numpy.ones(len(
                energies[sim_mode])) * job.statepoint.density

        if 'md' in sim_mode and not sim_mode.startswith('langevin'):
            momentum_vector = get_log_quantity(traj,
                                               'md/Integrator/linear_momentum')
            linear_momentum[sim_mode] = [
                math.sqrt(v[0]**2 + v[1]**2 + v[2]**2) for v in momentum_vector
            ]
        else:
            linear_momentum[sim_mode] = numpy.zeros(len(energies[sim_mode]))

        if 'md' in sim_mode:
            kinetic_temperature[sim_mode] = get_log_quantity(
                traj, 'md/compute/ThermodynamicQuantities/kinetic_temperature')
        else:
            kinetic_temperature[sim_mode] = numpy.ones(len(
                energies[sim_mode])) * job.statepoint.kT

    # save averages
    for mode in sim_modes:
        job.document[mode] = dict(
            pressure=float(numpy.mean(pressures[mode][-FRAMES_ANALYZE:])),
            potential_energy=float(numpy.mean(
                energies[mode][-FRAMES_ANALYZE:])),
            density=float(numpy.mean(densities[mode][-FRAMES_ANALYZE:])))

    # Plot results
    fig = matplotlib.figure.Figure(figsize=(20, 20 / 3.24 * 2), layout='tight')
    ax = fig.add_subplot(3, 2, 1)

    for mode in sim_modes:
        ax.plot(densities[mode], label=mode)
        ax.set_xlabel('frame')
        ax.set_ylabel(r'$\rho$')
        ax.legend()

    ax.hlines(y=job.statepoint.density,
              xmin=0,
              xmax=len(densities[sim_modes[0]]),
              linestyles='dashed',
              colors='k')

    ax = fig.add_subplot(3, 2, 2)
    for mode in sim_modes:
        ax.plot(pressures[mode], label=mode)
        ax.set_xlabel('frame')
        ax.set_ylabel('$P$')

    ax.hlines(y=job.statepoint.pressure,
              xmin=0,
              xmax=len(densities[sim_modes[0]]),
              linestyles='dashed',
              colors='k')

    ax = fig.add_subplot(3, 2, 3)
    for mode in sim_modes:
        ax.plot(numpy.array(energies[mode]) / job.statepoint.num_particles,
                label=mode)
        ax.set_xlabel('frame')
        ax.set_ylabel('$U / N$')

    ax = fig.add_subplot(3, 2, 4)
    for mode in sim_modes:
        ax.plot(kinetic_temperature[mode], label=mode)
        ax.set_xlabel('frame')
        ax.set_ylabel('kinetic temperature')

    ax.hlines(y=job.statepoint.kT,
              xmin=0,
              xmax=len(densities[sim_modes[0]]),
              linestyles='dashed',
              colors='k')

    ax = fig.add_subplot(3, 2, 5)
    for mode in sim_modes:
        ax.plot(numpy.array(linear_momentum[mode])
                / job.statepoint.num_particles,
                label=mode)
        ax.set_xlabel('frame')
        ax.set_ylabel(r'$|\vec{p}| / N$')

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

    ax = fig.add_subplot(3, 4, 11)
    max_density_histogram = 0
    for mode in sim_modes:
        density_histogram, bin_edges = numpy.histogram(
            densities[mode][-FRAMES_ANALYZE:], bins=100, range=density_range)
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

    ax = fig.add_subplot(3, 4, 12)
    max_pressure_histogram = 0
    for mode in sim_modes:
        pressure_histogram, bin_edges = numpy.histogram(
            pressures[mode][-FRAMES_ANALYZE:], bins=100, range=pressure_range)
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

    fig.suptitle(f"$kT={job.statepoint.kT}$, $\\rho={job.statepoint.density}$, "
                 f"$N={job.statepoint.num_particles}$, "
                 f"replicate={job.statepoint.replicate_idx}")
    fig.savefig(job.fn('plots.svg'), bbox_inches='tight')

    job.document['analysis_complete'] = True


def true_all(*jobs, key):
    """Check that a given key is true in all jobs."""
    return all(job.get(key, False) for job in jobs)


@aggregator.groupby(key=['kT', 'density', 'num_particles'],
                    sort_by='replicate_idx')
@LJFluid.operation.with_directives(
    directives=dict(executable=CONFIG["executable"]))
@LJFluid.pre(lambda *jobs: true_all(*jobs, key='analysis_complete'))
@LJFluid.post(lambda *jobs: true_all(*jobs, key='compare_modes_complete'))
def compare_modes(*jobs):
    """Compares the tested simulation modes."""
    import numpy
    import matplotlib
    import matplotlib.style
    import matplotlib.figure
    import scipy.stats
    matplotlib.style.use('ggplot')

    sim_modes = [
        'langevin_md_cpu', 'langevin_md_gpu', 'nvt_md_cpu', 'nvt_md_gpu',
        'npt_md_cpu', 'npt_md_gpu', 'nvt_mc_cpu', 'nvt_mc_gpu', 'npt_mc_cpu'
    ]
    quantity_names = ['density', 'pressure', 'potential_energy']

    # grab the common statepoint parameters
    kT = jobs[0].sp.kT
    set_density = jobs[0].sp.density
    set_pressure = jobs[0].sp.pressure
    num_particles = jobs[0].sp.num_particles

    quantity_reference = dict(density=set_density,
                              pressure=set_pressure,
                              potential_energy=None)

    fig = matplotlib.figure.Figure(figsize=(8, 8 / 1.618 * 3), layout='tight')
    fig.suptitle(f"$kT={kT}$, $\\rho={set_density}$, $N={num_particles}$")

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

        # Remove nan values, then run ANOVA test
        if quantity_name == 'pressure' and 'nvt_mc_cpu' in quantities:
            del quantities['nvt_mc_cpu']
        if quantity_name == 'pressure' and 'nvt_mc_gpu' in quantities:
            del quantities['nvt_mc_gpu']
        unpacked_quantities = list(quantities.values())
        f, p = scipy.stats.f_oneway(*unpacked_quantities)

        if p > 0.05:
            result = "$\\checkmark$"
        else:
            result = "XX"

        ax.set_title(label=result + f' ANOVA p-value: {p:0.3f}')

    filename = f'compare_kT{kT}_density{round(set_density, 2)}.svg'
    fig.savefig(os.path.join(jobs[0]._project.path, filename),
                bbox_inches='tight')

    for job in jobs:
        job.document['compare_modes_complete'] = True


if __name__ == "__main__":
    LJFluid.get_project(test_project_dict["LJFluid"].path).main()
