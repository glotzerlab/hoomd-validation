# Copyright (c) 2022-2023 The Regents of the University of Michigan.
# Part of HOOMD-blue, released under the BSD 3-Clause License.

"""Lennard Jones phase behavior validation test (union particles)."""

from config import CONFIG
from project_class import Project
from flow import aggregator
import util
import os
import math
import collections
import json
import pathlib

# Run parameters shared between simulations.
# Step counts must be even and a multiple of the log quantity period.
RANDOMIZE_STEPS = 20_000
EQUILIBRATE_STEPS = 100_000
RUN_STEPS = 500_000
RESTART_STEPS = RUN_STEPS // 10
TOTAL_STEPS = RANDOMIZE_STEPS + EQUILIBRATE_STEPS + RUN_STEPS

WRITE_PERIOD = 4_000
LOG_PERIOD = {'trajectory': 50_000, 'quantities': 100}
LJ_PARAMS = {'epsilon': 0.25, 'sigma': 1.0, 'r_on': 2.0, 'r_cut': 2.5}
NUM_CPU_RANKS = min(8, CONFIG["max_cores_sim"])
CUBE_VERTS = [
    (-0.5, -0.5, -0.5),
    (-0.5, -0.5, 0.5),
    (-0.5, 0.5, -0.5),
    (-0.5, 0.5, 0.5),
    (0.5, -0.5, -0.5),
    (0.5, -0.5, 0.5),
    (0.5, 0.5, -0.5),
    (0.5, 0.5, 0.5),
]

WALLTIME_STOP_SECONDS = CONFIG["max_walltime"] * 3600 - 10 * 60

# Limit the number of long NVE runs to reduce the number of CPU hours needed.
NUM_NVE_RUNS = 2


def job_statepoints():
    """list(dict): A list of statepoints for this subproject."""
    num_particles = 8**3
    replicate_indices = range(CONFIG["replicates"])
    params_list = [(1.5, 0.04, 0.10676024636823918)]
    for kT, density, pressure in params_list:
        for idx in replicate_indices:
            yield ({
                "subproject": "lj_union",
                "kT": kT,
                "density": density,
                "pressure": pressure,
                "num_particles": num_particles,
                "replicate_idx": idx
            })


def is_lj_union(job):
    """Test if a given job is part of the lj_union subproject."""
    return job.statepoint['subproject'] == 'lj_union'


partition_jobs_cpu_mpi = aggregator.groupsof(num=min(
    CONFIG["replicates"], CONFIG["max_cores_submission"] // NUM_CPU_RANKS),
                                             sort_by='density',
                                             select=is_lj_union)

partition_jobs_gpu = aggregator.groupsof(num=min(CONFIG["replicates"],
                                                 CONFIG["max_gpus_submission"]),
                                         sort_by='density',
                                         select=is_lj_union)


@Project.post.isfile('lj_union_initial_state_md.gsd')
@Project.operation(directives=dict(
    executable=CONFIG["executable"],
    nranks=util.total_ranks_function(NUM_CPU_RANKS),
    walltime=CONFIG['short_walltime']),
                   aggregator=partition_jobs_cpu_mpi)
def lj_union_create_initial_state(*jobs):
    """Create initial system configuration."""
    import hoomd
    import numpy
    import itertools

    min_spacing = math.sqrt(3) + 1

    communicator = hoomd.communicator.Communicator(
        ranks_per_partition=NUM_CPU_RANKS)
    job = jobs[communicator.partition]

    if communicator.rank == 0:
        print('starting lj_union_create_initial_state:', job)

    sp = job.sp
    device = hoomd.device.CPU(communicator=communicator,
                              message_filename=util.get_message_filename(
                                  job, 'create_initial_state.log'))

    box_volume = sp["num_particles"] / sp["density"]
    L = box_volume**(1 / 3.)

    N = int(numpy.ceil(sp["num_particles"]**(1. / 3.)))
    x = numpy.linspace(-L / 2, L / 2, N, endpoint=False)

    if x[1] - x[0] < min_spacing:
        print(x[1] - x[0], min_spacing)
        raise RuntimeError('density too high to initialize on cubic lattice')

    position = list(itertools.product(x, repeat=3))[:sp["num_particles"]]

    # create snapshot
    snap = hoomd.Snapshot(device.communicator)

    if device.communicator.rank == 0:
        snap.particles.N = sp["num_particles"]
        snap.particles.types = ['A', 'R']
        snap.configuration.box = [L, L, L, 0, 0, 0]
        snap.particles.position[:] = position
        snap.particles.moment_inertia[:] = (1, 1, 1)
        snap.particles.typeid[:] = [1] * sp["num_particles"]

    # Use hard sphere Monte-Carlo to randomize the initial configuration
    mc = hoomd.hpmc.integrate.Sphere()
    mc.shape['A'] = dict(diameter=0)
    mc.shape['R'] = dict(diameter=min_spacing, orientable=True)

    sim = hoomd.Simulation(device=device, seed=util.make_seed(job))
    sim.create_state_from_snapshot(snap)
    sim.operations.integrator = mc

    device.notice('Randomizing initial state...')
    sim.run(RANDOMIZE_STEPS)
    device.notice(f'Move counts: {mc.translate_moves}')
    device.notice('Done.')

    hoomd.write.GSD.write(state=sim.state,
                          filename=job.fn("lj_union_initial_state.gsd"),
                          mode='wb')

    # Create rigid bodies for MD
    sim.integrator = hoomd.md.Integrator(dt=0)
    rigid = hoomd.md.constrain.Rigid()
    rigid.body['R'] = {
        "constituent_types": ['A'] * 8,
        "positions": CUBE_VERTS,
        "orientations": [(1.0, 0.0, 0.0, 0.0)] * 8,
    }
    sim.integrator.rigid = rigid
    rigid.create_bodies(sim.state)

    hoomd.write.GSD.write(state=sim.state,
                          filename=job.fn("lj_union_initial_state_md.gsd"),
                          mode='wb')

    if communicator.rank == 0:
        print(f'completed lj_union_create_initial_state: {job}')


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
    nlist = md.nlist.Cell(buffer=0.4, exclusions=('body',))
    lj = md.pair.LJ(default_r_cut=LJ_PARAMS['r_cut'],
                    default_r_on=LJ_PARAMS['r_on'],
                    nlist=nlist)
    lj.params[('A', 'A')] = dict(sigma=LJ_PARAMS['sigma'],
                                 epsilon=LJ_PARAMS['epsilon'])

    lj.params[('A', 'R')] = dict(sigma=0, epsilon=0)
    lj.r_cut[('A', 'R')] = 0

    lj.params[('R', 'R')] = dict(sigma=0, epsilon=0)
    lj.r_cut[('R', 'R')] = 0

    lj.mode = 'xplor'

    # integrator
    integrator = md.Integrator(dt=0.0005,
                               methods=[method],
                               forces=[lj],
                               integrate_rotational_dof=True)

    # rigid bodies
    rigid = hoomd.md.constrain.Rigid()
    rigid.body['R'] = {
        "constituent_types": ['A'] * 8,
        "positions": CUBE_VERTS,
        "orientations": [(1.0, 0.0, 0.0, 0.0)] * 8,
    }
    integrator.rigid = rigid

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
                   'rotational_kinetic_energy',
                   'translational_kinetic_energy',
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


def run_md_sim(job, device, ensemble, thermostat, complete_filename):
    """Run the MD simulation with the given ensemble and thermostat."""
    import hoomd
    from hoomd import md
    from custom_actions import ComputeDensity

    initial_state = job.fn('lj_union_initial_state_md.gsd')

    integrate_filter = hoomd.filter.Rigid(flags=('center',))

    if ensemble == 'nvt':
        if thermostat == 'langevin':
            method = md.methods.Langevin(filter=integrate_filter,
                                         kT=job.statepoint.kT)
            method.gamma.default = 1.0
        elif thermostat == 'mttk':
            method = md.methods.ConstantVolume(filter=integrate_filter)
            method.thermostat = hoomd.md.methods.thermostats.MTTK(
                kT=job.statepoint.kT, tau=0.25)
        elif thermostat == 'bussi':
            method = md.methods.ConstantVolume(filter=integrate_filter)
            method.thermostat = hoomd.md.methods.thermostats.Bussi(
                kT=job.statepoint.kT)
        else:
            raise ValueError(f'Unsupported thermostat {thermostat}')
    elif ensemble == 'npt':
        p = job.statepoint.pressure
        method = md.methods.ConstantPressure(integrate_filter,
                                             S=[p, p, p, 0, 0, 0],
                                             tauS=3,
                                             couple='xyz')
        if thermostat == 'bussi':
            method.thermostat = hoomd.md.methods.thermostats.Bussi(
                kT=job.statepoint.kT)
        else:
            raise ValueError(f'Unsupported thermostat {thermostat}')

    sim_mode = f'{ensemble}_{thermostat}_md'

    density_compute = ComputeDensity(job.statepoint['num_particles'])
    sim = make_md_simulation(job,
                             device,
                             initial_state,
                             method,
                             sim_mode,
                             extra_loggables=[density_compute])

    # thermalize momenta
    sim.state.thermalize_particle_momenta(hoomd.filter.Rigid(flags=('center',)),
                                          job.sp.kT)

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

    pathlib.Path(job.fn(complete_filename)).touch()


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

    @Project.pre.after(lj_union_create_initial_state)
    @Project.post.isfile(f'{sim_mode}_{device_name}_complete')
    @Project.operation(name=f'lj_union_{sim_mode}_{device_name}',
                       directives=directives,
                       aggregator=aggregator)
    def md_sampling_operation(*jobs):
        """Perform sampling simulation given the definition."""
        import hoomd

        communicator = hoomd.communicator.Communicator(
            ranks_per_partition=ranks_per_partition)
        job = jobs[communicator.partition]

        if communicator.rank == 0:
            print(f'starting lj_union_{sim_mode}_{device_name}:', job)

        if device_name == 'gpu':
            device_cls = hoomd.device.GPU
        elif device_name == 'cpu':
            device_cls = hoomd.device.CPU

        device = device_cls(communicator=communicator,
                            message_filename=util.get_message_filename(
                                job, f'{sim_mode}_{device_name}.log'))

        run_md_sim(job,
                   device,
                   ensemble,
                   thermostat,
                   complete_filename=f'{sim_mode}_{device_name}_complete')

        if communicator.rank == 0:
            print(f'completed lj_union_{sim_mode}_{device_name}: {job}')

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
    from custom_actions import ComputeDensity

    # integrator
    mc = hpmc.integrate.Sphere(nselect=1)
    mc.shape['A'] = dict(diameter=0.0)
    mc.shape['R'] = dict(diameter=0.0, orientable=True)

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

    if isinstance(device, hoomd.device.CPU):
        code_isotropic = 'return 0.0f;'
    else:
        code_isotropic = ''

    lj_jit_potential = hpmc.pair.user.CPPPotentialUnion(
        r_cut_constituent=r_cut,
        code_constituent=lj_str,
        r_cut_isotropic=0,
        code_isotropic=code_isotropic,
        param_array_constituent=[],
        param_array_isotropic=[])
    lj_jit_potential.positions['A'] = []
    lj_jit_potential.diameters['A'] = []
    lj_jit_potential.typeids['A'] = []
    lj_jit_potential.orientations['A'] = []
    lj_jit_potential.charges['A'] = []

    lj_jit_potential.positions['R'] = CUBE_VERTS
    lj_jit_potential.diameters['R'] = [0] * len(CUBE_VERTS)
    lj_jit_potential.typeids['R'] = [0] * len(CUBE_VERTS)
    lj_jit_potential.orientations['R'] = [(1, 0, 0, 0)] * len(CUBE_VERTS)
    lj_jit_potential.charges['R'] = [0] * len(CUBE_VERTS)
    mc.pair_potential = lj_jit_potential

    # Note: Computing the virial pressure is non-trivial, skip

    # compute the density
    compute_density = ComputeDensity()

    # log to gsd
    logger = hoomd.logging.Logger(categories=['scalar', 'sequence'])
    logger.add(lj_jit_potential, quantities=['energy'])
    logger.add(mc, quantities=['translate_moves', 'rotate_moves'])
    logger.add(compute_density)
    for loggable in extra_loggables:
        logger.add(loggable)

    # make simulation
    sim = util.make_simulation(job=job,
                               device=device,
                               initial_state=initial_state,
                               integrator=mc,
                               sim_mode=sim_mode,
                               logger=logger,
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

    # move size tuner
    mstuner = hpmc.tune.MoveSize.scale_solver(
        moves=['a', 'd'],
        types=['R'],
        target=0.2,
        max_translation_move=0.5,
        max_rotation_move=1.0,
        trigger=hoomd.trigger.And([
            hoomd.trigger.Periodic(100),
            hoomd.trigger.Before(RANDOMIZE_STEPS | EQUILIBRATE_STEPS // 2)
        ]))
    sim.operations.add(mstuner)

    return sim


def run_nvt_mc_sim(job, device, complete_filename):
    """Run MC sim in NVT."""
    import hoomd

    if not hoomd.version.llvm_enabled:
        device.notice("LLVM disabled, skipping MC simulations.")
        return

    # simulation

    sim_mode = 'nvt_mc'
    restart_filename = util.get_job_filename(sim_mode, device, 'restart', 'gsd')
    if job.isfile(restart_filename):
        initial_state = job.fn(restart_filename)
        restart = True
    else:
        initial_state = job.fn('lj_union_initial_state.gsd')
        restart = False

    sim = make_mc_simulation(job, device, initial_state, sim_mode)

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
            f'Translate trial move size: {sim.operations.integrator.d["R"]}')

        rotate_moves = sim.operations.integrator.rotate_moves
        rotate_acceptance = rotate_moves[0] / sum(rotate_moves)
        device.notice(f'Rotate move acceptance: {rotate_acceptance}')
        device.notice(
            f'Rotate trial move size: {sim.operations.integrator.a["R"]}')

        # save move size to a file
        if device.communicator.rank == 0:
            name = util.get_job_filename(sim_mode, device, 'move_size', 'json')
            with open(job.fn(name), 'w') as f:
                json.dump(
                    dict(
                        d_R=sim.operations.integrator.d["R"],
                        a_R=sim.operations.integrator.a["R"],
                    ), f)
    else:
        device.notice('Restarting...')
        # read move size from the file
        name = util.get_job_filename(sim_mode, device, 'move_size', 'json')
        with open(job.fn(name), 'r') as f:
            data = json.load(f)

        sim.operations.integrator.d["R"] = data['d_R']
        sim.operations.integrator.a["R"] = data['a_R']
        device.notice(
            f'Restored translate move size: {sim.operations.integrator.d["R"]}')
        device.notice(
            f'Restored rotate move size: {sim.operations.integrator.a["R"]}')

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


def run_npt_mc_sim(job, device, complete_filename):
    """Run MC sim in NPT."""
    import hoomd
    from hoomd import hpmc

    if not hoomd.version.llvm_enabled:
        device.notice("LLVM disabled, skipping MC simulations.")
        return

    sim_mode = 'npt_mc'

    restart_filename = util.get_job_filename(sim_mode, device, 'restart', 'gsd')
    if job.isfile(restart_filename):
        initial_state = job.fn(restart_filename)
        restart = True
    else:
        initial_state = job.fn('lj_union_initial_state.gsd')
        restart = False

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
            f'Translate trial move size: {sim.operations.integrator.d["R"]}')

        rotate_moves = sim.operations.integrator.rotate_moves
        rotate_acceptance = rotate_moves[0] / sum(rotate_moves)
        device.notice(f'Rotate move acceptance: {rotate_acceptance}')
        device.notice(
            f'Rotate trial move size: {sim.operations.integrator.a["R"]}')

        volume_moves = boxmc.volume_moves
        volume_acceptance = volume_moves[0] / sum(volume_moves)
        device.notice(f'Volume move acceptance: {volume_acceptance}')
        device.notice(f'Volume move size: {boxmc.volume["delta"]}')

        # save move sizes to a file
        if device.communicator.rank == 0:
            name = util.get_job_filename(sim_mode, device, 'move_size', 'json')
            with open(job.fn(name), 'w') as f:
                json.dump(
                    dict(d_R=sim.operations.integrator.d["R"],
                         a_R=sim.operations.integrator.a["R"],
                         volume_delta=boxmc.volume['delta']), f)
    else:
        device.notice('Restarting...')
        # read move size from the file
        name = util.get_job_filename(sim_mode, device, 'move_size', 'json')
        with open(job.fn(name), 'r') as f:
            data = json.load(f)

        sim.operations.integrator.d["R"] = data['d_R']
        sim.operations.integrator.a["R"] = data['a_R']
        device.notice(
            f'Restored translate move size: {sim.operations.integrator.d["R"]}')
        device.notice(
            f'Restored rotate move size: {sim.operations.integrator.a["R"]}')
        boxmc.volume = dict(weight=1.0, mode='ln', delta=data['volume_delta'])
        device.notice(f'Restored volume move size: {boxmc.volume["delta"]}')

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

    @Project.pre.after(lj_union_create_initial_state)
    @Project.post.isfile(f'{mode}_mc_{device_name}_complete')
    @Project.operation(name=f'lj_union_{mode}_mc_{device_name}',
                       directives=directives,
                       aggregator=aggregator)
    def sampling_operation(*jobs):
        """Perform sampling simulation given the definition."""
        import hoomd

        communicator = hoomd.communicator.Communicator(
            ranks_per_partition=ranks_per_partition)
        job = jobs[communicator.partition]

        if communicator.rank == 0:
            print(f'starting lj_union_{mode}_mc_{device_name}:', job)

        if device_name == 'gpu':
            device_cls = hoomd.device.GPU
        elif device_name == 'cpu':
            device_cls = hoomd.device.CPU

        device = device_cls(communicator=communicator,
                            message_filename=util.get_message_filename(
                                job, f'{mode}_mc_{device_name}.log'))

        globals().get(f'run_{mode}_mc_sim')(
            job, device, complete_filename=f'{mode}_mc_{device_name}_complete')

        if communicator.rank == 0:
            print(f'completed lj_union_{mode}_mc_{device_name} {job}')

    mc_sampling_jobs.append(sampling_operation)


if CONFIG['enable_llvm']:
    for definition in mc_job_definitions:
        add_mc_sampling_job(**definition)


@Project.pre(is_lj_union)
@Project.pre.after(*md_sampling_jobs)
@Project.pre.after(*mc_sampling_jobs)
@Project.post.true('lj_union_analysis_complete')
@Project.operation(directives=dict(walltime=CONFIG['short_walltime'],
                                   executable=CONFIG["executable"]))
def lj_union_analyze(job):
    """Analyze the output of all simulation modes."""
    import h5py
    import numpy
    import math
    import matplotlib
    import matplotlib.style
    import matplotlib.figure
    matplotlib.style.use('fivethirtyeight')

    print('starting lj_union_analyze:', job)

    sim_modes = [
        'nvt_langevin_md_cpu',
        'nvt_mttk_md_cpu',
        'nvt_bussi_md_cpu',
        'npt_bussi_md_cpu',
    ]

    if os.path.exists(job.fn('nvt_langevin_md_gpu_quantities.h5')):
        sim_modes.extend([
            'nvt_langevin_md_gpu',
            'nvt_mttk_md_gpu',
            'nvt_bussi_md_gpu',
            'npt_bussi_md_gpu',
        ])

    if os.path.exists(job.fn('nvt_mc_cpu_quantities.h5')):
        sim_modes.extend(['nvt_mc_cpu', 'npt_mc_cpu'])

    if os.path.exists(job.fn('nvt_mc_gpu_quantities.h5')):
        sim_modes.extend(['nvt_mc_gpu'])

    util._sort_sim_modes(sim_modes)

    timesteps = {}
    energies = {}
    pressures = {}
    densities = {}
    linear_momentum = {}

    for sim_mode in sim_modes:
        log_traj = h5py.File(mode='r', name=job.fn(sim_mode + '_quantities.h5'))

        timesteps[sim_mode] = log_traj['hoomd-data/Simulation/timestep']

        if 'md' in sim_mode:
            energies[sim_mode] = log_traj[
                'hoomd-data/md/compute/ThermodynamicQuantities/potential_energy']
        else:
            energies[sim_mode] = (
                log_traj['hoomd-data/hpmc/pair/user/CPPPotentialUnion/energy']
                * job.statepoint.kT)

        energies[sim_mode] /= job.statepoint.num_particles

        if 'md' in sim_mode:
            pressures[sim_mode] = log_traj[
                'hoomd-data/md/compute/ThermodynamicQuantities/pressure']
        else:
            pressures[sim_mode] = numpy.full(len(energies[sim_mode]), numpy.nan)

        densities[sim_mode] = log_traj[
            'hoomd-data/custom_actions/ComputeDensity/density']

        if 'md' in sim_mode and 'langevin' not in sim_mode:
            momentum_vector = log_traj[
                'hoomd-data/md/Integrator/linear_momentum']
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

    fig = matplotlib.figure.Figure(figsize=(20, 20 / 3.24 * 2), layout='tight')
    ax = fig.add_subplot(2, 2, 1)
    util.plot_timeseries(ax=ax,
                         timesteps=timesteps,
                         data=densities,
                         ylabel=r"$\rho$",
                         expected=job.sp.density,
                         max_points=500)
    ax.legend()

    ax = fig.add_subplot(2, 2, 2)
    util.plot_timeseries(ax=ax,
                         timesteps=timesteps,
                         data=pressures,
                         ylabel=r"$P$",
                         expected=job.sp.pressure,
                         max_points=500)

    ax = fig.add_subplot(2, 2, 3)
    util.plot_timeseries(ax=ax,
                         timesteps=timesteps,
                         data=energies,
                         ylabel="$U / N$",
                         max_points=500)

    ax = fig.add_subplot(2, 2, 4)
    util.plot_timeseries(ax=ax,
                         timesteps=timesteps,
                         data={
                             mode: numpy.asarray(lm) / job.sp.num_particles
                             for mode, lm in linear_momentum.items()
                         },
                         ylabel=r'$|\vec{p}| / N$',
                         max_points=500)

    fig.suptitle(f"$kT={job.statepoint.kT}$, $\\rho={job.statepoint.density}$, "
                 f"$N={job.statepoint.num_particles}$, "
                 f"replicate={job.statepoint.replicate_idx}")
    fig.savefig(job.fn('nvt_npt_plots.svg'), bbox_inches='tight')

    job.document['lj_union_analysis_complete'] = True


analysis_aggregator = aggregator.groupby(key=['kT', 'density', 'num_particles'],
                                         sort_by='replicate_idx',
                                         select=is_lj_union)


@Project.pre(
    lambda *jobs: util.true_all(*jobs, key='lj_union_analysis_complete'))
@Project.post(
    lambda *jobs: util.true_all(*jobs, key='lj_union_compare_modes_complete'))
@Project.operation(directives=dict(walltime=CONFIG['short_walltime'],
                                   executable=CONFIG["executable"]),
                   aggregator=analysis_aggregator)
def lj_union_compare_modes(*jobs):
    """Compares the tested simulation modes."""
    import numpy
    import matplotlib
    import matplotlib.style
    import matplotlib.figure
    matplotlib.style.use('fivethirtyeight')

    print('starting lj_union_compare_modes:', jobs[0])

    sim_modes = [
        'nvt_langevin_md_cpu',
        'nvt_mttk_md_cpu',
        'nvt_bussi_md_cpu',
        'npt_bussi_md_cpu',
    ]

    if os.path.exists(jobs[0].fn('nvt_langevin_md_gpu_quantities.h5')):
        sim_modes.extend([
            'nvt_langevin_md_gpu',
            'nvt_mttk_md_gpu',
            'nvt_bussi_md_gpu',
            'npt_bussi_md_gpu',
        ])

    if os.path.exists(jobs[0].fn('nvt_mc_cpu_quantities.h5')):
        sim_modes.extend(['nvt_mc_cpu', 'npt_mc_cpu'])

    if os.path.exists(jobs[0].fn('nvt_mc_gpu_quantities.h5')):
        sim_modes.extend(['nvt_mc_gpu'])

    util._sort_sim_modes(sim_modes)

    quantity_names = ['density', 'pressure', 'potential_energy']

    labels = {
        'density': r'$\frac{\rho_\mathrm{sample} - \rho}{\rho} \cdot 1000$',
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

    fig = matplotlib.figure.Figure(figsize=(10, 10 / 1.618 * 2), layout='tight')
    fig.suptitle(f"$kT={kT}$, $\\rho={set_density}$, $N={num_particles}$")

    for i, quantity_name in enumerate(quantity_names):
        ax = fig.add_subplot(3, 1, i + 1)

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

        if quantity_name == "density":
            if 'npt_mc_cpu' in avg_quantity:
                print(f"Average npt_mc_cpu density {num_particles}:",
                      avg_quantity['npt_mc_cpu'], '+/-',
                      stderr_quantity['npt_mc_cpu'])
            print(f"Average npt_md_cpu density {num_particles}:",
                  avg_quantity['npt_bussi_md_cpu'], '+/-',
                  stderr_quantity['npt_bussi_md_cpu'])

    filename = f'lj_union_compare_kT{kT}_density{round(set_density, 2)}.svg'
    fig.savefig(os.path.join(jobs[0]._project.path, filename),
                bbox_inches='tight')

    for job in jobs:
        job.document['lj_union_compare_modes_complete'] = True


@Project.pre.after(*md_sampling_jobs)
@Project.post(lambda *jobs: util.true_all(
    *jobs, key='lj_union_distribution_analyze_complete'))
@Project.operation(directives=dict(walltime=CONFIG['short_walltime'],
                                   executable=CONFIG["executable"]),
                   aggregator=analysis_aggregator)
def lj_union_distribution_analyze(*jobs):
    """Checks that MD follows the correct KE distribution."""
    import h5py
    import numpy
    import matplotlib
    import matplotlib.style
    import matplotlib.figure
    import scipy
    matplotlib.style.use('fivethirtyeight')

    print('starting lj_union_distribution_analyze:', jobs[0])

    sim_modes = [
        'nvt_langevin_md_cpu',
        'nvt_mttk_md_cpu',
        'nvt_bussi_md_cpu',
        'npt_bussi_md_cpu',
    ]

    if os.path.exists(jobs[0].fn('nvt_langevin_md_gpu_quantities.h5')):
        sim_modes.extend([
            'nvt_langevin_md_gpu',
            'nvt_mttk_md_gpu',
            'nvt_bussi_md_gpu',
            'npt_bussi_md_gpu',
        ])

    if os.path.exists(jobs[0].fn('nvt_mc_cpu_quantities.h5')):
        sim_modes.extend(['nvt_mc_cpu', 'npt_mc_cpu'])

    if os.path.exists(jobs[0].fn('nvt_mc_gpu_quantities.h5')):
        sim_modes.extend(['nvt_mc_gpu'])

    util._sort_sim_modes(sim_modes)

    # grab the common statepoint parameters
    kT = jobs[0].sp.kT
    set_density = jobs[0].sp.density
    num_particles = jobs[0].sp.num_particles

    fig = matplotlib.figure.Figure(figsize=(20, 20 / 3.24 * 4), layout='tight')
    fig.suptitle(f"$kT={kT}$, $\\rho={set_density}$, $N={num_particles}$")

    # n_dof_translate = num_particles * 3 - 3
    # n_dof_rotate = num_particles * 3
    # n_dof_total = num_particles * 6 - 3

    ke_translate_means_expected = collections.defaultdict(list)
    ke_translate_sigmas_expected = collections.defaultdict(list)
    ke_translate_samples = collections.defaultdict(list)

    ke_rotate_means_expected = collections.defaultdict(list)
    ke_rotate_sigmas_expected = collections.defaultdict(list)
    ke_rotate_samples = collections.defaultdict(list)

    potential_energy_samples = collections.defaultdict(list)
    pressure_samples = collections.defaultdict(list)
    density_samples = collections.defaultdict(list)

    for job in jobs:
        for sim_mode in sim_modes:
            if sim_mode.startswith('nvt_langevin'):
                n_translate_dof = num_particles * 3
            else:
                n_translate_dof = num_particles * 3 - 3

            n_rotate_dof = num_particles * 3

            print('Reading' + job.fn(sim_mode + '_quantities.h5'))
            log_traj = h5py.File(mode='r',
                                 name=job.fn(sim_mode + '_quantities.h5'))

            if 'md' in sim_mode:
                # https://doi.org/10.1371/journal.pone.0202764
                ke_translate = log_traj[
                    'hoomd-data/md/compute/ThermodynamicQuantities/'
                    'translational_kinetic_energy']
                ke_translate_means_expected[sim_mode].append(
                    numpy.mean(ke_translate) - 1 / 2 * n_translate_dof * kT)
                ke_translate_sigmas_expected[sim_mode].append(
                    numpy.std(ke_translate)
                    - 1 / math.sqrt(2) * math.sqrt(n_translate_dof) * kT)
                ke_translate_samples[sim_mode].extend(ke_translate)

                ke_rotate = log_traj[
                    'hoomd-data/md/compute/ThermodynamicQuantities/'
                    'rotational_kinetic_energy']
                ke_rotate_means_expected[sim_mode].append(
                    numpy.mean(ke_rotate) - 1 / 2 * n_rotate_dof * kT)
                ke_rotate_sigmas_expected[sim_mode].append(
                    numpy.std(ke_rotate)
                    - 1 / math.sqrt(2) * math.sqrt(n_rotate_dof) * kT)
                ke_rotate_samples[sim_mode].extend(ke_rotate)
            else:
                ke_translate_samples[sim_mode].extend(
                    [3 / 2 * job.statepoint.num_particles * job.statepoint.kT])
                ke_rotate_samples[sim_mode].extend(
                    [3 / 2 * job.statepoint.num_particles * job.statepoint.kT])

            if 'md' in sim_mode:
                potential_energy_samples[sim_mode].extend(
                    log_traj['hoomd-data/md/compute/ThermodynamicQuantities'
                             '/potential_energy'])
            else:
                potential_energy_samples[sim_mode].extend(log_traj[
                    'hoomd-data/hpmc/pair/user/CPPPotentialUnion/energy']
                                                          * job.statepoint.kT)

            if 'md' in sim_mode:
                pressure_samples[sim_mode].extend(log_traj[
                    'hoomd-data/md/compute/ThermodynamicQuantities/pressure'])
            else:
                pressure_samples[sim_mode].extend([job.statepoint.pressure])

            density_samples[sim_mode].extend(
                log_traj['hoomd-data/custom_actions/ComputeDensity/density'])

    ax = fig.add_subplot(4, 2, 1)
    util.plot_vs_expected(ax, ke_translate_means_expected,
                          r'$<K_\mathrm{translate}> - 1/2 N_{dof} k T$')

    ax = fig.add_subplot(4, 2, 2)
    util.plot_vs_expected(
        ax, ke_translate_sigmas_expected,
        r'$\Delta K_\mathrm{translate} - 1/\sqrt{2} \sqrt{N_{dof}} k T$')

    ax = fig.add_subplot(4, 2, 3)
    util.plot_vs_expected(ax, ke_rotate_means_expected,
                          r'$<K_\mathrm{rotate}> - 1/2 N_{dof} k T$')

    ax = fig.add_subplot(4, 2, 4)
    util.plot_vs_expected(
        ax, ke_rotate_sigmas_expected,
        r'$\Delta K_\mathrm{rotate} - 1/\sqrt{2} \sqrt{N_{dof}} k T$')

    ax = fig.add_subplot(4, 2, 5)
    rv = scipy.stats.gamma(3 * job.statepoint.num_particles / 2,
                           scale=job.statepoint.kT)
    util.plot_distribution(ax,
                           ke_translate_samples,
                           r'$K_\mathrm{translate}$',
                           expected=rv.pdf)
    ax.legend(loc='upper right', fontsize='xx-small')

    ax = fig.add_subplot(4, 2, 6)
    util.plot_distribution(ax,
                           ke_rotate_samples,
                           r'$K_\mathrm{rotate}$',
                           expected=rv.pdf)

    ax = fig.add_subplot(4, 4, 13)
    util.plot_distribution(ax, potential_energy_samples, 'U')

    ax = fig.add_subplot(4, 4, 14)
    util.plot_distribution(ax,
                           density_samples,
                           r'$\rho$',
                           expected=job.statepoint.density)

    ax = fig.add_subplot(4, 4, 15)
    util.plot_distribution(ax,
                           pressure_samples,
                           'P',
                           expected=job.statepoint.pressure)

    filename = f'lj_union_distribution_analyze_kT{kT}_' \
               f'density{round(set_density, 2)}.svg'
    fig.savefig(os.path.join(jobs[0]._project.path, filename),
                bbox_inches='tight')

    for job in jobs:
        job.document['lj_union_distribution_analyze_complete'] = True


#################################
# MD conservation simulations
#################################


def run_nve_md_sim(job, device, run_length, complete_filename):
    """Run the MD simulation in NVE."""
    import hoomd

    sim_mode = 'nve_md'
    restart_filename = util.get_job_filename(sim_mode, device, 'restart', 'gsd')
    is_restarting = job.isfile(restart_filename)

    if is_restarting:
        initial_state = job.fn(restart_filename)
    else:
        initial_state = job.fn('lj_union_initial_state_md.gsd')

    nve = hoomd.md.methods.ConstantVolume(hoomd.filter.Rigid(flags=('center',)))

    sim = make_md_simulation(job,
                             device,
                             initial_state,
                             nve,
                             sim_mode,
                             period_multiplier=200)

    if not is_restarting:
        sim.state.thermalize_particle_momenta(
            hoomd.filter.Rigid(flags=('center',)), job.sp.kT)

    # Run for a long time to look for energy and momentum drift
    device.notice('Running...')

    util.run_up_to_walltime(sim=sim,
                            end_step=RANDOMIZE_STEPS + EQUILIBRATE_STEPS
                            + run_length,
                            steps=500_000,
                            walltime_stop=WALLTIME_STOP_SECONDS)

    if sim.timestep == RANDOMIZE_STEPS + EQUILIBRATE_STEPS + run_length:
        pathlib.Path(job.fn(complete_filename)).touch()
        device.notice('Done.')
    else:
        device.notice('Ending run early due to walltime limits at:'
                      f'{device.communicator.walltime}')

    hoomd.write.GSD.write(state=sim.state,
                          filename=job.fn(restart_filename),
                          mode='wb')


def is_lj_union_nve(job):
    """Test if a given job should be run for NVE conservation."""
    return job.statepoint['subproject'] == 'lj_union' and \
        job.statepoint['replicate_idx'] < NUM_NVE_RUNS


partition_jobs_cpu_mpi_nve = aggregator.groupsof(num=min(
    CONFIG["replicates"], CONFIG["max_cores_submission"] // NUM_CPU_RANKS),
                                                 sort_by='density',
                                                 select=is_lj_union_nve)

partition_jobs_gpu_nve = aggregator.groupsof(num=min(
    CONFIG["replicates"], CONFIG["max_gpus_submission"]),
                                             sort_by='density',
                                             select=is_lj_union_nve)

nve_md_sampling_jobs = []
nve_md_job_definitions = [
    {
        'device_name': 'cpu',
        'ranks_per_partition': NUM_CPU_RANKS,
        'aggregator': partition_jobs_cpu_mpi_nve,
        'run_length': 10_000_000,
    },
]

if CONFIG["enable_gpu"]:
    nve_md_job_definitions.extend([
        {
            'device_name': 'gpu',
            'ranks_per_partition': 1,
            'aggregator': partition_jobs_gpu_nve,
            'run_length': 100_000_000,
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

    @Project.pre.after(lj_union_create_initial_state)
    @Project.post.isfile(f'{sim_mode}_{device_name}_complete')
    @Project.operation(name=f'lj_union_{sim_mode}_{device_name}',
                       directives=directives,
                       aggregator=aggregator)
    def lj_union_nve_md_job(*jobs):
        """Run NVE MD."""
        import hoomd

        communicator = hoomd.communicator.Communicator(
            ranks_per_partition=ranks_per_partition)
        job = jobs[communicator.partition]

        if communicator.rank == 0:
            print(f'starting lj_union_{sim_mode}_{device_name}:', job)

        if device_name == 'gpu':
            device_cls = hoomd.device.GPU
        elif device_name == 'cpu':
            device_cls = hoomd.device.CPU

        device = device_cls(communicator=communicator,
                            message_filename=util.get_message_filename(
                                job, f'{sim_mode}_{device_name}.log'))
        run_nve_md_sim(job,
                       device,
                       run_length=run_length,
                       complete_filename=f'{sim_mode}_{device_name}_complete')

        if communicator.rank == 0:
            print(f'completed lj_union_{sim_mode}_{device_name} {job}')

    nve_md_sampling_jobs.append(lj_union_nve_md_job)


for definition in nve_md_job_definitions:
    add_nve_md_job(**definition)

nve_analysis_aggregator = aggregator.groupby(
    key=['kT', 'density', 'num_particles'],
    sort_by='replicate_idx',
    select=is_lj_union_nve)


@Project.pre.after(*nve_md_sampling_jobs)
@Project.post(lambda *jobs: util.true_all(
    *jobs, key='lj_union_conservation_analysis_complete'))
@Project.operation(directives=dict(walltime=CONFIG['short_walltime'],
                                   executable=CONFIG["executable"]),
                   aggregator=nve_analysis_aggregator)
def lj_union_conservation_analyze(*jobs):
    """Analyze the output of NVE simulations and inspect conservation."""
    import h5py
    import math
    import matplotlib
    import matplotlib.style
    import matplotlib.figure
    matplotlib.style.use('fivethirtyeight')

    print('starting lj_union_conservation_analyze:', jobs[0])

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
            log_traj = h5py.File(mode='r',
                                 name=job.fn(sim_mode + '_quantities.h5'))

            job_timesteps[sim_mode] = log_traj['hoomd-data/Simulation/timestep']

            job_energies[sim_mode] = (
                log_traj[
                    'hoomd-data/md/compute/ThermodynamicQuantities/potential_energy']
                + log_traj[
                    'hoomd-data/md/compute/ThermodynamicQuantities/kinetic_energy']
            )
            job_energies[sim_mode] = (
                job_energies[sim_mode]
                - job_energies[sim_mode][0]) / job.statepoint["num_particles"]

            momentum_vector = log_traj[
                'hoomd-data/md/Integrator/linear_momentum']
            job_linear_momentum[sim_mode] = [
                math.sqrt(v[0]**2 + v[1]**2 + v[2]**2)
                / job.statepoint["num_particles"] for v in momentum_vector
            ]

        timesteps.append(job_timesteps)
        energies.append(job_energies)
        linear_momenta.append(job_linear_momentum)

    # Plot results
    def plot(*, ax, data, quantity_name, legend=False):
        for i, job in enumerate(jobs):
            for mode in sim_modes:
                ax.plot(timesteps[i][mode],
                        data[i][mode],
                        label=f'{mode}_{job.statepoint.replicate_idx}')
        ax.set_xlabel("time step")
        ax.set_ylabel(quantity_name)

        if legend:
            ax.legend()

    fig = matplotlib.figure.Figure(figsize=(10, 10 / 1.68 * 2), layout='tight')
    ax = fig.add_subplot(2, 1, 1)
    plot(ax=ax, data=energies, quantity_name=r"$E / N$", legend=True)

    ax = fig.add_subplot(2, 1, 2)
    plot(ax=ax,
         data=linear_momenta,
         quantity_name=r"$\left| \vec{p} \right| / N$")

    fig.suptitle("LJ union conservation tests: "
                 f"$kT={job.statepoint.kT}$, $\\rho={job.statepoint.density}$, "
                 f"$N={job.statepoint.num_particles}$")
    filename = f'lj_union_conservation_kT{job.statepoint.kT}_' \
               f'density{round(job.statepoint.density, 2)}_' \
               f'N{job.statepoint.num_particles}.svg'

    fig.savefig(os.path.join(jobs[0]._project.path, filename),
                bbox_inches='tight')

    for job in jobs:
        job.document['lj_union_conservation_analysis_complete'] = True
