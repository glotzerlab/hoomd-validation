# First make the project just run, then put in project file

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
            r_cut=4.0,
            r_on=3.2,
        ),
        dict(
            kT=1.0,
            density=0.9193740949934834,
            pressure=11.0,
            num_particles=12**3,
            r_cut=2**(1 / 6),
            r_on=2.0,
        ),
    ]

    for param in params_list:
        for idx in replicate_indices:
            yield ({
                "subproject": "patchylj_fluid",
                "kT": param['kT'],
                "density": param['density'],
                "pressure": param['pressure'],
                "num_particles": param['num_particles'],
                "replicate_idx": idx,
                "r_cut": param['r_cut'],
                "r_on": param['r_on'],
            })


def is_patchylj_fluid(job):
    """Test if a given job is part of the patchylj_fluid subproject."""
    return job.statepoint['subproject'] == 'patchylj_fluid'

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
    lj = md.pair.LJ(default_r_cut=job.statepoint.r_cut,
                    default_r_on=job.statepoint.r_on,
                    nlist=nlist)
    lj.params[('A', 'A')] = dict(sigma=LJ_PARAMS['sigma'],
                                 epsilon=LJ_PARAMS['epsilon'])
    lj.mode = 'xplor'

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
    r_on = job.statepoint.r_on
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
                          default_r_on=job.statepoint.r_on,
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
