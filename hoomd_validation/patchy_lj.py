

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
LJ_PARAMS = {'epsilon': 1.0, 'sigma': 1.0}
NUM_CPU_RANKS = min(8, CONFIG["max_cores_sim"])

WALLTIME_STOP_SECONDS = CONFIG["max_walltime"] * 3600 - 10 * 60


# TODO copied from lj_fluid for now
def job_statepoints():
    """list(dict): A list of statepoints for this subproject."""
    replicate_indices = range(CONFIG["replicates"])
    params_list = [
        dict(
            kT=1.5,
            density=0.6269137133228043,
            pressure=1.0,
            num_particles=16**3,
            r_cut=4.0
        ),
        dict(
            kT=1.0,
            density=0.9193740949934834,
            pressure=11.0,
            num_particles=12**3,
            r_cut=2**(1 / 6)
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
                "r_cut": param['r_cut']
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
                                             select=is_lj_fluid)

partition_jobs_gpu = aggregator.groupsof(num=min(CONFIG["replicates"],
                                                 CONFIG["max_gpus_submission"]),
                                         sort_by=sort_key,
                                         select=is_lj_fluid)


# TODO copied from lj_fluid, modify if needed
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
    wca = md.pair.LJ(default_r_cut=job.statepoint.r_cut,
                    nlist=nlist)
    wca.params[('A', 'A')] = dict(sigma=LJ_PARAMS['sigma'],
                                 epsilon=LJ_PARAMS['epsilon'])
    wca.mode = 'shift'

    # integrator
    integrator = md.Integrator(dt=0.001, methods=[method], forces=[lj])

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
    r_cut = job.statepoint.r_cut

    # WCA goes to 0 at 2^1/6 sigma, no XPLOR needed

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
                    vec3 ni_world = rotate(q_i, n_i);
                    vec3 nj_world = rotate(q_j, n_j);

                    float magdr = sqrt(rsq);
                    vec3 rhat = r_ij / magdr;

                    float costhetai = -dot(rhat, ni_world);
                    float costhetaj = dot(rhat, nj_world);

                    float fi()
                    {{
                        return 1.0f / (1.0f + exp(-{omega} * (costhetai - {cosalpha})) );
                    }}
                    float fj()
                    {{
                        return 1.0f / (1.0f + exp(-{omega} * (costhetaj - {cosalpha})) );
                    }}

    
                    // loop over patches eventually
                    float this_envelope = fi() * fj();
 
                    // regular lj to be modulated
                    r_cut = {r_cut};
                    r_cutsq = r_cut * r_cut;
                    
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
                        float lj_energy = 4 * {epsilon} * (r12inv - r6inv);
                        
                        // energy shift at cutoff
                        float r_cutsqinv = sigsq / r_cutsq;
                        r_cut6inv = r_cutsqinv * r_cutsqinv * r_cutsqinv;
                        r_cut12inv = r_cut6inv * r_cut6inv;
                        float cutVal = 4 * {epsilon} * (r_cut12inv - r_cut6inv);
                        lj_energy -= cutVal;
                    }}
                   
                    return wca_energy + lj_energy * this_envelope;
                    """.format(epsilon=epsilon, sigma=sigma, r_cut=r_cut,
                               wca_epsilon=epsilon, wca_sigma=sigma, r_cut_wca=sigma * 2**(1/6),
                               omega=omega, cosalpha=numpy.cos(alpha))

    jit_potential = hpmc.pair.user.CPPPotential(r_cut=r_cut,
                                                code=patchy_lj_str,
                                                param_array=[])
    mc.pair_potential = jit_potential


    # pair force to compute virial pressure
    nlist = hoomd.md.nlist.Cell(buffer=0.4)
    wca = hoomd.md.pair.LJ(default_r_cut=sigma * 2**(1/6),
                          nlist=nlist)
    wca.params[('A', 'A')] = dict(sigma=LJ_PARAMS['sigma'],
                                 epsilon=LJ_PARAMS['epsilon']) # TODO: why not /kt
    wca.mode = 'shift'

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
        return job.statepoint.num_particles * job.statepoint.kT / V + w / (3
                                                                           * V)

    logger_gsd[('custom', 'virial_pressure')] = (_compute_virial_pressure,
                                                 'scalar')

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








        
# HPMC with the patchy potential

# params are alpha, omega, patches

patchyLJCode = """



"""


patch = hoomd.hpmc.pair.user.CPPPotential(r_cut = 5,
                                          code = patchyLJCode,
                                          param_array = )
