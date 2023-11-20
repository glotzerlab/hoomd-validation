

"""Patchy LJ validation test."""

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
WCA_PARAMS = {'epsilon': 1.0, 'sigma': 1.0}
NUM_CPU_RANKS = min(1, CONFIG["max_cores_sim"])

WALLTIME_STOP_SECONDS = CONFIG["max_walltime"] * 3600 - 10 * 60

# Limit the number of long NVE runs to reduce the number of CPU hours needed.
NUM_NVE_RUNS = 2

def job_statepoints():
    """list(dict): A list of statepoints for this subproject."""
    import numpy
    replicate_indices = range(CONFIG["replicates"])
    params_list = [
        dict(
            kT=1.0,
            density=0.9193740949934834,
            pressure=11.0,
            num_particles=7**3,
            r_cut=3.0,
            patch_vector=(1,0,0), # make sure it is normalized here
            alpha=numpy.deg2rad(20),
            omega=20,
            patch_epsilon=2,
            patch_sigma=1
        ),
    ]

    for param in params_list:
        for idx in replicate_indices:
            yield ({
                "subproject": "patchy_lj_fluid",
                "kT": param['kT'],
                "density": param['density'],
                "pressure": param['pressure'],
                "num_particles": param['num_particles'],
                "replicate_idx": idx,
                "r_cut": param['r_cut'],
                "patch_vector": param['patch_vector'],
                "alpha": param['alpha'],
                "omega": param['omega'],
                "patch_sigma": param['patch_sigma'],
                "patch_epsilon": param['patch_epsilon']
            })


def is_patchy_lj_fluid(job):
    """Test if a given job is part of the patchy_lj_fluid subproject."""
    return job.statepoint['subproject'] == 'patchy_lj_fluid'

def sort_key(job):
    """Aggregator sort key."""
    return (job.statepoint.density, job.statepoint.num_particles)

partition_jobs_cpu_mpi = aggregator.groupsof(num=min(
    CONFIG["replicates"], CONFIG["max_cores_submission"] // NUM_CPU_RANKS),
                                             sort_by=sort_key,
                                             select=is_patchy_lj_fluid)

partition_jobs_gpu = aggregator.groupsof(num=min(CONFIG["replicates"],
                                                 CONFIG["max_gpus_submission"]),
                                         sort_by=sort_key,
                                         select=is_patchy_lj_fluid)


@Project.post.isfile('patchy_lj_fluid_initial_state.gsd')
@Project.operation(directives=dict(
    executable=CONFIG["executable"],
    nranks=util.total_ranks_function(NUM_CPU_RANKS),
    walltime=CONFIG['short_walltime']),
                   aggregator=partition_jobs_cpu_mpi)
def patchy_lj_fluid_create_initial_state(*jobs):
    """Create initial system configuration."""
    import hoomd
    import numpy
    import itertools

    communicator = hoomd.communicator.Communicator(
        ranks_per_partition=NUM_CPU_RANKS)
    job = jobs[communicator.partition]

    if communicator.rank == 0:
        print('starting patchy_lj_fluid_create_initial_state:', job)

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
    device.notice(f'Move counts: {mc.translate_moves}')
    device.notice('Done.')

    hoomd.write.GSD.write(state=sim.state,
                          filename=job.fn("patchy_lj_fluid_initial_state.gsd"),
                          mode='wb')

    if communicator.rank == 0:
        print(f'completed patchy_lj_fluid_create_initial_state: '
              f'{job} in {communicator.walltime} s')


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
    r_cut_wca = WCA_PARAMS['sigma'] * 2**(1/6)
    wca = md.pair.LJ(default_r_cut=r_cut_wca,
                     nlist=nlist)
    wca.params[('A', 'A')] = dict(sigma=WCA_PARAMS['sigma'],
                                 epsilon=WCA_PARAMS['epsilon'])
    wca.mode = 'shift'

    patch = md.pair.aniso.PatchyLJ(nlist=nlist,
                                   default_r_cut=job.statepoint.r_cut)
    patch.patches['A'] = [job.statepoint.patch_vector]
    patch.params[('A', 'A')] = dict(pair_params=dict(epsilon=job.statepoint.patch_epsilon,
                                                     sigma=job.statepoint.patch_sigma),
                                    envelope_params=dict(alpha=job.statepoint.alpha,
                                                         omega = job.statepoint.omega))

    # integrator
    integrator = md.Integrator(dt=0.001, methods=[method], forces=[wca, patch],
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

    initial_state = job.fn('patchy_lj_fluid_initial_state.gsd')

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

    pathlib.Path(job.fn(complete_filename)).touch()
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

    @Project.pre.after(patchy_lj_fluid_create_initial_state)
    @Project.post.isfile(f'{sim_mode}_{device_name}_complete')
    @Project.operation(name=f'patchy_lj_fluid_{sim_mode}_{device_name}',
                       directives=directives,
                       aggregator=aggregator)
    def md_sampling_operation(*jobs):
        """Perform sampling simulation given the definition."""
        import hoomd

        communicator = hoomd.communicator.Communicator(
            ranks_per_partition=ranks_per_partition)
        job = jobs[communicator.partition]

        if communicator.rank == 0:
            print(f'starting patchy_lj_fluid_{sim_mode}_{device_name}:', job)

        if device_name == 'gpu':
            device_cls = hoomd.device.GPU
        elif device_name == 'cpu':
            device_cls = hoomd.device.CPU

        device = device_cls(
            communicator=communicator,
            message_filename=job.fn(f'{sim_mode}_{device_name}.log'))

        run_md_sim(job,
                   device,
                   ensemble,
                   thermostat,
                   complete_filename=f'{sim_mode}_{device_name}_complete')

        if communicator.rank == 0:
            print(f'completed patchy_lj_fluid_{sim_mode}_{device_name}: '
                  f'{job} in {communicator.walltime} s')

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
    from hoomd import md
    import numpy
    from custom_actions import ComputeDensity

    # integrator
    mc = hpmc.integrate.Sphere(nselect=1)
    mc.shape['A'] = dict(diameter=0.0, orientable=True)

    # pair potential
    epsilon = job.statepoint.patch_epsilon / job.sp.kT
    sigma = job.statepoint.patch_sigma
    wca_epsilon = WCA_PARAMS['epsilon'] / job.sp.kT  # noqa F841
    wca_sigma = WCA_PARAMS['sigma']
    r_cut = job.statepoint.r_cut # not for WCA
    omega = job.statepoint.omega
    alpha = job.statepoint.alpha

    nx,ny,nz = job.statepoint.patch_vector
    
    patchy_lj_str = """
                // WCA via shift

                float sigma;
                float sigsq;
                float rsqinv;
                float r6inv;
                float r12inv;
                float wca_energy;

                float rsq = dot(r_ij, r_ij);
                float r_cut = {r_cut_wca};
                float r_cutsq = r_cut * r_cut;
          
                if (rsq >= r_cutsq)
                {{
                    wca_energy = 0.0f;
                }}
                else
                {{
                    sigma = {wca_sigma};
                    sigsq = sigma * sigma;
                    rsqinv = sigsq / rsq;
                    r6inv = rsqinv * rsqinv * rsqinv;
                    r12inv = r6inv * r6inv;
                    wca_energy = 4 * {wca_epsilon} * (r12inv - r6inv);
                   
                    // energy shift for WCA
                    wca_energy += {wca_epsilon};
                }}

                // patchy stuff

                vec3<float> n_i({nx},{ny},{nz});
                vec3<float> n_j({nx},{ny},{nz});
                vec3<float> ni_world = rotate(q_i, n_i);
                vec3<float> nj_world = rotate(q_j, n_j);

                float magdr = sqrt(rsq);
                vec3<float> rhat = r_ij / magdr;

                float costhetai = -dot(rhat, ni_world);
                float costhetaj = dot(rhat, nj_world);


                // loop over patches eventually
                float this_envelope = (1.0f / (1.0f + exp(-{omega} * (costhetai - {cosalpha})) ) ) * (1.0f / (1.0f + exp(-{omega} * (costhetaj - {cosalpha})) ) );

                // regular lj to be modulated
                r_cut = {r_cut};
                r_cutsq = r_cut * r_cut;

                float lj_energy;

                if (rsq >= r_cutsq)
                {{
                    lj_energy = 0.0f;
                }}
                else
                {{
                    sigma = {sigma};
                    sigsq = sigma * sigma;
                    rsqinv = sigsq / rsq;
                    r6inv = rsqinv * rsqinv * rsqinv;
                    r12inv = r6inv * r6inv;
                    lj_energy = 4 * {epsilon} * (r12inv - r6inv);
                    
                    // energy shift at cutoff
                    float r_cutsqinv = sigsq / r_cutsq;
                    float r_cut6inv = r_cutsqinv * r_cutsqinv * r_cutsqinv;
                    float r_cut12inv = r_cut6inv * r_cut6inv;
                    float cutVal = 4 * {epsilon} * (r_cut12inv - r_cut6inv);
                    lj_energy -= cutVal;
                }}
               
                return wca_energy + lj_energy * this_envelope;
                """.format(epsilon=epsilon, sigma=sigma, r_cut=r_cut,
                           wca_epsilon=wca_epsilon, wca_sigma=wca_sigma, r_cut_wca=wca_sigma * 2**(1/6),
                           omega=omega, cosalpha=numpy.cos(alpha),
                           nx=nx, ny=ny, nz=nz)

    jit_potential = hpmc.pair.user.CPPPotential(r_cut=r_cut,
                                                code=patchy_lj_str,
                                                param_array=[])
    mc.pair_potential = jit_potential


    # pair forces to compute virial pressure
    nlist = hoomd.md.nlist.Cell(buffer=0.4)
    wca = hoomd.md.pair.LJ(default_r_cut=wca_sigma * 2**(1/6),
                           nlist=nlist)
    wca.params[('A', 'A')] = dict(sigma=WCA_PARAMS['sigma'],
                                  epsilon=wca_epsilon) # TODO: why not /kt WCA_PARAMS['epsilon']
    wca.mode = 'shift'

    patch = md.pair.aniso.PatchyLJ(nlist=nlist,
                                   default_r_cut = job.statepoint.r_cut)
    patch.patches['A'] = [job.statepoint.patch_vector]
    patch.params[('A', 'A')] = dict(pair_params=dict(epsilon=job.statepoint.patch_epsilon,
                                                     sigma=job.statepoint.patch_sigma),
                                    envelope_params=dict(alpha=job.statepoint.alpha,
                                                         omega = job.statepoint.omega))

    # compute the density
    compute_density = ComputeDensity()

    # log to gsd
    logger_gsd = hoomd.logging.Logger(categories=['scalar', 'sequence'])
    logger_gsd.add(jit_potential, quantities=['energy'])
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
        virials = numpy.sum(wca.virials, 0) # TODO should be MD patch potential object
        virials2 = numpy.sum(patch.virials, 0)
        w = 0
        if virials is not None:
            w = virials[0] + virials[3] + virials[5]
        if virials2 is not None:
            w += virials2[0] + virials2[3] + virials2[5]
        V = sim.state.box.volume
        return job.statepoint.num_particles * job.statepoint.kT / V + w / (3
                                                                           * V)

    logger_gsd[('custom', 'virial_pressure')] = (_compute_virial_pressure,
                                                 'scalar')

    # move size tuner
    mstuner = hpmc.tune.MoveSize.scale_solver(
        moves=['d', 'a'],
        target=0.2,
        max_translation_move=0.5,
        max_rotation_move=1.0,
        trigger=hoomd.trigger.And([
            hoomd.trigger.Periodic(100),
            hoomd.trigger.Before(RANDOMIZE_STEPS | EQUILIBRATE_STEPS // 2)
        ]))
    sim.operations.add(mstuner)
    sim.operations.computes.append(wca)
    sim.operations.computes.append(patch)

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
        initial_state = job.fn('patchy_lj_fluid_initial_state.gsd')
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
        device.notice(f'Trial move size: {sim.operations.integrator.d["A"]}')

        rotate_moves = sim.operations.integrator.rotate_moves
        rotate_acceptance = rotate_moves[0] / sum(rotate_moves)
        device.notice(f'Rotate move acceptance: {rotate_acceptance}')
        device.notice(
            f'Rotate move size: {sim.operations.integrator.a["A"]}')

        # save move size to a file
        if device.communicator.rank == 0:
            name = util.get_job_filename(sim_mode, device, 'move_size', 'json')
            with open(job.fn(name), 'w') as f:
                json.dump(
                    dict(d_A=sim.operations.integrator.d["A"],
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
        device.notice(
            f'Restored trial move size: {sim.operations.integrator.d["A"]}')
        device.notice(
            f'Restored rotate move size: {sim.operations.integrator.a["A"]}')

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

    # device
    sim_mode = 'npt_mc'
    restart_filename = util.get_job_filename(sim_mode, device, 'restart', 'gsd')
    if job.isfile(restart_filename):
        initial_state = job.fn(restart_filename)
        restart = True
    else:
        initial_state = job.fn('patchy_lj_fluid_initial_state.gsd')
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
        device.notice(f'Trial move size: {sim.operations.integrator.d["A"]}')

        rotate_moves = sim.operations.integrator.rotate_moves
        rotate_acceptance = rotate_moves[0] / sum(rotate_moves)
        device.notice(f'Rotate move acceptance: {rotate_acceptance}')
        device.notice(
            f'Rotate move size: {sim.operations.integrator.a["A"]}')

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
                         volume_delta=boxmc.volume['delta']
                    ), f)
    else:
        device.notice('Restarting...')
        # read move size from the file
        name = util.get_job_filename(sim_mode, device, 'move_size', 'json')
        with open(job.fn(name), 'r') as f:
            data = json.load(f)

        sim.operations.integrator.d["A"] = data['d_A']
        sim.operations.integrator.a["A"] = data['a_A']
        boxmc.volume = dict(weight=1.0, mode='ln', delta=data['volume_delta'])
        device.notice(
            f'Restored trial move size: {sim.operations.integrator.d["A"]}')
        device.notice(
            f'Restored rotate move size: {sim.operations.integrator.a["A"]}')
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

    @Project.pre.after(patchy_lj_fluid_create_initial_state)
    @Project.post.isfile(f'{mode}_mc_{device_name}_complete')
    @Project.operation(name=f'patchy_lj_fluid_{mode}_mc_{device_name}',
                       directives=directives,
                       aggregator=aggregator)
    def sampling_operation(*jobs):
        """Perform sampling simulation given the definition."""
        import hoomd

        communicator = hoomd.communicator.Communicator(
            ranks_per_partition=ranks_per_partition)
        job = jobs[communicator.partition]

        if communicator.rank == 0:
            print(f'starting patchy_lj_fluid_{mode}_mc_{device_name}:', job)

        if device_name == 'gpu':
            device_cls = hoomd.device.GPU
        elif device_name == 'cpu':
            device_cls = hoomd.device.CPU

        device = device_cls(
            communicator=communicator,
            message_filename=job.fn(f'{mode}_mc_{device_name}.log'))

        globals().get(f'run_{mode}_mc_sim')(
            job, device, complete_filename=f'{mode}_mc_{device_name}_complete')

        if communicator.rank == 0:
            print(f'completed patchy_lj_fluid_{mode}_mc_{device_name} '
                  f'{job} in {communicator.walltime} s')

    mc_sampling_jobs.append(sampling_operation)


if CONFIG['enable_llvm']:
    for definition in mc_job_definitions:
        add_mc_sampling_job(**definition)


@Project.pre(is_patchy_lj_fluid)
@Project.pre.after(*md_sampling_jobs)
@Project.pre.after(*mc_sampling_jobs)
@Project.post.true('patchy_lj_fluid_analysis_complete')
@Project.operation(directives=dict(walltime=CONFIG['short_walltime'],
                                   executable=CONFIG["executable"]))
def patchy_lj_fluid_analyze(job):
    """Analyze the output of all simulation modes."""
    import gsd.hoomd
    import numpy
    import math
    import matplotlib
    import matplotlib.style
    import matplotlib.figure
    matplotlib.style.use('fivethirtyeight')

    print('starting patchy_lj_fluid_analyze:', job)

    sim_modes = [
        'nvt_langevin_md_cpu',
        'nvt_mttk_md_cpu',
        'nvt_bussi_md_cpu',
        'npt_bussi_md_cpu',
    ]

    if os.path.exists(job.fn('nvt_langevin_md_gpu_quantities.gsd')):
        sim_modes.extend([
            'nvt_langevin_md_gpu',
            'nvt_mttk_md_gpu',
            'nvt_bussi_md_gpu',
            'npt_bussi_md_gpu',
        ])

    if os.path.exists(job.fn('nvt_mc_cpu_quantities.gsd')):
        sim_modes.extend(['nvt_mc_cpu', 'npt_mc_cpu'])

    if os.path.exists(job.fn('nvt_mc_gpu_quantities.gsd')):
        sim_modes.extend(['nvt_mc_gpu'])

    util._sort_sim_modes(sim_modes)

    timesteps = {}
    energies = {}
    pressures = {}
    densities = {}
    linear_momentum = {}

    for sim_mode in sim_modes:
        log_traj = gsd.hoomd.read_log(job.fn(sim_mode + '_quantities.gsd'))

        timesteps[sim_mode] = log_traj['configuration/step']

        if 'md' in sim_mode:
            energies[sim_mode] = log_traj[
                'log/md/compute/ThermodynamicQuantities/potential_energy']
        else:
            energies[sim_mode] = log_traj[
                'log/hpmc/pair/user/CPPPotential/energy'] * job.statepoint.kT

        energies[sim_mode] /= job.statepoint.num_particles

        if 'md' in sim_mode:
            pressures[sim_mode] = log_traj[
                'log/md/compute/ThermodynamicQuantities/pressure']
        else:
            pressures[sim_mode] = log_traj['log/custom/virial_pressure']

        densities[sim_mode] = log_traj[
            'log/custom_actions/ComputeDensity/density']

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
                 f"$r_\\mathrm{{cut}}={job.statepoint.r_cut}$, "
                 f"replicate={job.statepoint.replicate_idx}")
    fig.savefig(job.fn('nvt_npt_plots.svg'), bbox_inches='tight')

    job.document['patchy_lj_fluid_analysis_complete'] = True


analysis_aggregator = aggregator.groupby(
    key=['kT', 'density', 'num_particles', 'r_cut'],
    sort_by='replicate_idx',
    select=is_patchy_lj_fluid)


@Project.pre(
    lambda *jobs: util.true_all(*jobs, key='patchy_lj_fluid_analysis_complete'))
@Project.post(
    lambda *jobs: util.true_all(*jobs, key='patchy_lj_fluid_compare_modes_complete'))
@Project.operation(directives=dict(walltime=CONFIG['short_walltime'],
                                   executable=CONFIG["executable"]),
                   aggregator=analysis_aggregator)
def patchy_lj_fluid_compare_modes(*jobs):
    """Compares the tested simulation modes."""
    import numpy
    import matplotlib
    import matplotlib.style
    import matplotlib.figure
    matplotlib.style.use('fivethirtyeight')

    print('starting patchy_lj_fluid_compare_modes:', jobs[0])

    sim_modes = [
        'nvt_langevin_md_cpu',
        'nvt_mttk_md_cpu',
        'nvt_bussi_md_cpu',
        'npt_bussi_md_cpu',
    ]

    if os.path.exists(jobs[0].fn('nvt_langevin_md_gpu_quantities.gsd')):
        sim_modes.extend([
            'nvt_langevin_md_gpu',
            'nvt_mttk_md_gpu',
            'nvt_bussi_md_gpu',
            'npt_bussi_md_gpu',
        ])

    if os.path.exists(jobs[0].fn('nvt_mc_cpu_quantities.gsd')):
        sim_modes.extend(['nvt_mc_cpu', 'npt_mc_cpu'])

    if os.path.exists(jobs[0].fn('nvt_mc_gpu_quantities.gsd')):
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

    fig = matplotlib.figure.Figure(figsize=(10, 10 / 1.618 * 3), layout='tight')
    fig.suptitle(f"$kT={kT}$, $\\rho={set_density}$, "
                 f"$r_\\mathrm{{cut}}={jobs[0].statepoint.r_cut}$, "
                 f"$N={num_particles}$")

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
            print(f"Average npt_mc_cpu density {num_particles}:",
                  avg_quantity['npt_mc_cpu'], '+/-',
                  stderr_quantity['npt_mc_cpu'])
            print(f"Average npt_md_cpu density {num_particles}:",
                  avg_quantity['npt_bussi_md_cpu'], '+/-',
                  stderr_quantity['npt_bussi_md_cpu'])
        if quantity_name == "pressure":
            print(f"Average nvt_mc_cpu pressure {num_particles}:",
                  avg_quantity['nvt_mc_cpu'], '+/-',
                  stderr_quantity['nvt_mc_cpu'])
            print(f"Average npt_mc_cpu pressure {num_particles}:",
                  avg_quantity['npt_mc_cpu'], '+/-',
                  stderr_quantity['npt_mc_cpu'])

    filename = f'patchy_lj_fluid_compare_kT{kT}_density{round(set_density, 2)}_' \
               f'r_cut{round(jobs[0].statepoint.r_cut, 2)}_' \
               f'N{num_particles}.svg'

    fig.savefig(os.path.join(jobs[0]._project.path, filename),
                bbox_inches='tight')

    for job in jobs:
        job.document['patchy_lj_fluid_compare_modes_complete'] = True


@Project.pre.after(*md_sampling_jobs)
@Project.post(lambda *jobs: util.true_all(
    *jobs, key='patchy_lj_fluid_distribution_analyze_complete'))
@Project.operation(directives=dict(walltime=CONFIG['short_walltime'],
                                   executable=CONFIG["executable"]),
                   aggregator=analysis_aggregator)
def patchy_lj_fluid_distribution_analyze(*jobs):
    """Checks that MD follows the correct KE distribution."""
    import gsd.hoomd
    import numpy
    import matplotlib
    import matplotlib.style
    import matplotlib.figure
    import scipy
    matplotlib.style.use('fivethirtyeight')

    print('starting patchy_lj_fluid_distribution_analyze:', jobs[0])

    sim_modes = [
        'nvt_langevin_md_cpu',
        'nvt_mttk_md_cpu',
        'nvt_bussi_md_cpu',
        'npt_bussi_md_cpu',
    ]

    if os.path.exists(jobs[0].fn('nvt_langevin_md_gpu_quantities.gsd')):
        sim_modes.extend([
            'nvt_langevin_md_gpu',
            'nvt_mttk_md_gpu',
            'nvt_bussi_md_gpu',
            'npt_bussi_md_gpu',
        ])

    if os.path.exists(jobs[0].fn('nvt_mc_cpu_quantities.gsd')):
        sim_modes.extend(['nvt_mc_cpu', 'npt_mc_cpu'])

    if os.path.exists(jobs[0].fn('nvt_mc_gpu_quantities.gsd')):
        sim_modes.extend(['nvt_mc_gpu'])

    util._sort_sim_modes(sim_modes)

    # grab the common statepoint parameters
    kT = jobs[0].sp.kT
    set_density = jobs[0].sp.density
    num_particles = jobs[0].sp.num_particles

    fig = matplotlib.figure.Figure(figsize=(20, 20 / 3.24 * 2), layout='tight')
    fig.suptitle(f"$kT={kT}$, $\\rho={set_density}$, "
                 f"$r_\\mathrm{{cut}}={jobs[0].statepoint.r_cut}$, "
                 f"$N={num_particles}$")

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
                ke = log_traj[
                    'log/md/compute/ThermodynamicQuantities/kinetic_energy']
                ke_means_expected[sim_mode].append(
                    numpy.mean(ke) - 1 / 2 * n_dof * kT)
                ke_sigmas_expected[sim_mode].append(
                    numpy.std(ke) - 1 / math.sqrt(2) * math.sqrt(n_dof) * kT)

                ke_samples[sim_mode].extend(ke)
            else:
                ke_samples[sim_mode].extend(
                    [3 / 2 * job.statepoint.num_particles * job.statepoint.kT])

            if 'md' in sim_mode:
                potential_energy_samples[sim_mode].extend(
                    list(log_traj['log/md/compute/ThermodynamicQuantities'
                                  '/potential_energy']))
            else:
                potential_energy_samples[sim_mode].extend(
                    list(log_traj['log/hpmc/pair/user/CPPPotential/energy']
                         * job.statepoint.kT))

            if 'md' in sim_mode:
                pressure_samples[sim_mode].extend(
                    list(log_traj[
                        'log/md/compute/ThermodynamicQuantities/pressure']))
            else:
                pressure_samples[sim_mode].extend(
                    list(log_traj['log/custom/virial_pressure']))

            density_samples[sim_mode].extend(
                list(log_traj['log/custom_actions/ComputeDensity/density']))

    ax = fig.add_subplot(2, 2, 1)
    util.plot_vs_expected(ax, ke_means_expected, '$<K> - 1/2 N_{dof} k T$')

    ax = fig.add_subplot(2, 2, 2)
    # https://doi.org/10.1371/journal.pone.0202764
    util.plot_vs_expected(ax, ke_sigmas_expected,
                          r'$\Delta K - 1/\sqrt{2} \sqrt{N_{dof}} k T$')

    ax = fig.add_subplot(2, 4, 5)
    rv = scipy.stats.gamma(3 * job.statepoint.num_particles / 2,
                           scale=job.statepoint.kT)
    util.plot_distribution(ax, ke_samples, 'K', expected=rv.pdf)
    ax.legend(loc='upper right', fontsize='xx-small')

    ax = fig.add_subplot(2, 4, 6)
    util.plot_distribution(ax, potential_energy_samples, 'U')

    ax = fig.add_subplot(2, 4, 7)
    util.plot_distribution(ax,
                           density_samples,
                           r'$\rho$',
                           expected=job.statepoint.density)

    ax = fig.add_subplot(2, 4, 8)
    util.plot_distribution(ax,
                           pressure_samples,
                           'P',
                           expected=job.statepoint.pressure)

    filename = f'patchy_lj_fluid_distribution_analyze_kT{kT}'\
               f'_density{round(set_density, 2)}_' \
               f'r_cut{round(jobs[0].statepoint.r_cut, 2)}_' \
               f'N{num_particles}.svg'
    fig.savefig(os.path.join(jobs[0]._project.path, filename),
                bbox_inches='tight')

    for job in jobs:
        job.document['patchy_lj_fluid_distribution_analyze_complete'] = True

    
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
        initial_state = job.fn('patchy_lj_fluid_initial_state.gsd')

    nve = hoomd.md.methods.ConstantVolume(hoomd.filter.All())

    sim = make_md_simulation(job,
                             device,
                             initial_state,
                             nve,
                             sim_mode,
                             period_multiplier=200)

    if not is_restarting:
        sim.state.thermalize_particle_momenta(hoomd.filter.All(), job.sp.kT)

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


def is_patchy_lj_fluid_nve(job):
    """Test if a given job should be run for NVE conservation."""
    return job.statepoint['subproject'] == 'patchy_lj_fluid' and \
        job.statepoint['replicate_idx'] < NUM_NVE_RUNS


partition_jobs_cpu_mpi_nve = aggregator.groupsof(num=min(
    CONFIG["replicates"], CONFIG["max_cores_submission"] // NUM_CPU_RANKS),
                                                 sort_by=sort_key,
                                                 select=is_patchy_lj_fluid_nve)

partition_jobs_gpu_nve = aggregator.groupsof(num=min(
    CONFIG["replicates"], CONFIG["max_gpus_submission"]),
                                             sort_by=sort_key,
                                             select=is_patchy_lj_fluid_nve)

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

    @Project.pre.after(patchy_lj_fluid_create_initial_state)
    @Project.post.isfile(f'{sim_mode}_{device_name}_complete')
    @Project.operation(name=f'patchy_lj_fluid_{sim_mode}_{device_name}',
                       directives=directives,
                       aggregator=aggregator)
    def patchy_lj_fluid_nve_md_job(*jobs):
        """Run NVE MD."""
        import hoomd

        communicator = hoomd.communicator.Communicator(
            ranks_per_partition=ranks_per_partition)
        job = jobs[communicator.partition]

        if communicator.rank == 0:
            print(f'starting patchy_lj_fluid_{sim_mode}_{device_name}:', job)

        if device_name == 'gpu':
            device_cls = hoomd.device.GPU
        elif device_name == 'cpu':
            device_cls = hoomd.device.CPU

        device = device_cls(
            communicator=communicator,
            message_filename=job.fn(f'{sim_mode}_{device_name}.log'))
        run_nve_md_sim(job,
                       device,
                       run_length=run_length,
                       complete_filename=f'{sim_mode}_{device_name}_complete')

        if communicator.rank == 0:
            print(f'completed patchy_lj_fluid_{sim_mode}_{device_name} '
                  f'{job} in {communicator.walltime} s')

    nve_md_sampling_jobs.append(patchy_lj_fluid_nve_md_job)


for definition in nve_md_job_definitions:
    add_nve_md_job(**definition)

nve_analysis_aggregator = aggregator.groupby(
    key=['kT', 'density', 'num_particles', 'r_cut'],
    sort_by='replicate_idx',
    select=is_patchy_lj_fluid_nve)


@Project.pre.after(*nve_md_sampling_jobs)
@Project.post(lambda *jobs: util.true_all(
    *jobs, key='patchy_lj_fluid_conservation_analysis_complete'))
@Project.operation(directives=dict(walltime=CONFIG['short_walltime'],
                                   executable=CONFIG["executable"]),
                   aggregator=nve_analysis_aggregator)
def patchy_lj_fluid_conservation_analyze(*jobs):
    """Analyze the output of NVE simulations and inspect conservation."""
    import gsd.hoomd
    import numpy
    import math
    import matplotlib
    import matplotlib.style
    import matplotlib.figure
    matplotlib.style.use('fivethirtyeight')

    print('starting patchy_lj_fluid_conservation_analyze:', jobs[0])

    sim_modes = ['nve_md_cpu']
    if os.path.exists(jobs[0].fn('nve_md_gpu_quantities.gsd')):
        sim_modes.extend(['nve_md_gpu'])

    timesteps = []
    energies = []
    linear_momenta = []

    for job in jobs:
        job_timesteps = {}
        job_energies = {}
        job_linear_momentum = {}

        for sim_mode in sim_modes:
            log_traj = gsd.hoomd.read_log(job.fn(sim_mode + '_quantities.gsd'))

            job_timesteps[sim_mode] = log_traj['configuration/step']

            job_energies[sim_mode] = (
                log_traj[
                    'log/md/compute/ThermodynamicQuantities/potential_energy']
                + log_traj[
                    'log/md/compute/ThermodynamicQuantities/kinetic_energy'])
            job_energies[sim_mode] = (
                job_energies[sim_mode]
                - job_energies[sim_mode][0]) / job.statepoint["num_particles"]

            momentum_vector = log_traj['log/md/Integrator/linear_momentum']
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
                        numpy.asarray(data[i][mode]),
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

    fig.suptitle("LJ conservation tests: "
                 f"$kT={job.statepoint.kT}$, $\\rho={job.statepoint.density}$, "
                 f"$r_\\mathrm{{cut}}={job.statepoint.r_cut}$, "
                 f"$N={job.statepoint.num_particles}$")
    filename = f'patchy_lj_fluid_conservation_kT{job.statepoint.kT}_' \
               f'density{round(job.statepoint.density, 2)}_' \
               f'r_cut{round(jobs[0].statepoint.r_cut, 2)}_' \
               f'N{job.statepoint.num_particles}.svg'

    fig.savefig(os.path.join(jobs[0]._project.path, filename),
                bbox_inches='tight')

    for job in jobs:
        job.document['patchy_lj_fluid_conservation_analysis_complete'] = True
