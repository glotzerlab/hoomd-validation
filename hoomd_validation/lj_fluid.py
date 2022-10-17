# Copyright (c) 2022 The Regents of the University of Michigan.
# Part of HOOMD-blue, released under the BSD 3-Clause License.

"""Lennard Jones phase behavior validation test."""

import numpy as np
from config import test_project_dict, CONFIG
from project_classes import LJFluid
from flow import aggregator

# Run parameters shared between simulations
RANDOMIZE_STEPS = 1e5
RUN_STEPS = 2e6
WRITE_PERIOD = 1000
LOG_PERIOD = {'trajectory': 50000, 'quantities': 125}
FRAMES_ANALYZE = int(RUN_STEPS / LOG_PERIOD['quantities'] * 2/3)
LJ_PARAMS = {'epsilon': 1.0, 'sigma': 1.0, 'r_on': 2.0, 'r_cut': 2.5}


@LJFluid.operation(
    directives=dict(executable=CONFIG["executable"], nranks=8))
@LJFluid.post.isfile('initial_state.gsd')
def create_initial_state(job):
    """Create initial system configuration."""
    import hoomd
    import itertools

    sp = job.sp
    device = hoomd.device.CPU()

    box_volume = sp["num_particles"] / sp["density"]
    L = box_volume**(1 / 3.)

    N = int(np.ceil(sp["num_particles"]**(1. / 3.)))
    x = np.linspace(-L / 2, L / 2, N, endpoint=False)

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
    mc = hoomd.hpmc.integrate.Sphere(nselect=1)
    mc.shape['A'] = dict(diameter=1.0)

    sim = hoomd.Simulation(device=device, seed=job.statepoint.replicate_idx)
    sim.create_state_from_snapshot(snap)
    sim.operations.integrator = mc

    device.notice('Randomizing initial state...')
    sim.run(RANDOMIZE_STEPS)
    device.notice(f'Done. Move counts: {mc.translate_moves}')

    # equilibrate the initial configuration
    nlist = hoomd.md.nlist.Cell(buffer=0.4)
    lj = hoomd.md.pair.LJ(default_r_cut=LJ_PARAMS['r_cut'],
                    default_r_on=LJ_PARAMS['r_on'],
                    nlist=nlist)
    lj.params[('A', 'A')] = dict(sigma=LJ_PARAMS['sigma'],
                                 epsilon=LJ_PARAMS['epsilon'])
    lj.mode = 'xplor'

    # integrator
    langevin = hoomd.md.methods.Langevin(hoomd.filter.All(), kT=job.sp.kT)
    langevin.gamma.default = 1.0
    sim.operations.integrator = hoomd.md.Integrator(dt=0.00244140625, methods=[langevin], forces=[lj])

    device.notice(f'Equilibrating...')
    sim.run(RANDOMIZE_STEPS)
    device.notice(f'Done.')

    hoomd.write.GSD.write(state=sim.state, filename=job.fn("initial_state.gsd"), mode='wb')


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
        filename=job.fn(f"{sim_mode}_trajectory.gsd"),
        trigger=hoomd.trigger.Periodic(LOG_PERIOD['trajectory']),
        mode='wb')
    sim.operations.add(trajectory_writer)

    # write logged quantities to gsd file
    quantity_writer = hoomd.write.GSD(
        filter=hoomd.filter.Null(),
        filename=job.fn(f"{sim_mode}_quantities.gsd"),
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
    integrator = md.Integrator(dt=0.00244140625, methods=[method], forces=[lj])

    # compute thermo
    thermo = md.compute.ThermodynamicQuantities(hoomd.filter.All())

    # add gsd log quantities
    logger = hoomd.logging.Logger(categories=['scalar', 'sequence'])
    logger.add(thermo, quantities=['pressure', 'potential_energy', 'kinetic_temperature'])
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

@LJFluid.operation(directives=dict(
    walltime=48, executable=CONFIG["executable"], nranks=8))
@LJFluid.pre.after(create_initial_state)
@LJFluid.post.true('langevin_md_complete')
def run_langevin_md_sim(job):
    """Run the MD simulation in Langevin."""
    import hoomd
    from hoomd import md

    device = hoomd.device.CPU()
    initial_state = job.fn('initial_state.gsd')
    langevin = md.methods.Langevin(hoomd.filter.All(), kT=job.sp.kT)
    langevin.gamma.default = 1.0

    sim_mode = 'langevin_md'

    sim = make_md_simulation(job, device, initial_state, langevin, sim_mode)

    # run
    sim.run(RUN_STEPS)

    job.document['langevin_md_complete'] = True


@LJFluid.operation(directives=dict(
    walltime=48, executable=CONFIG["executable"], nranks=8))
@LJFluid.pre.after(create_initial_state)
@LJFluid.post.true('nvt_md_complete')
def run_nvt_md_sim(job):
    """Run the MD simulation in NVT."""
    import hoomd
    from hoomd import md

    device = hoomd.device.CPU()
    initial_state = job.fn('initial_state.gsd')
    nvt = md.methods.NVT(hoomd.filter.All(), kT=job.sp.kT, tau=0.24)
    sim_mode = 'nvt_md'

    sim = make_md_simulation(job, device, initial_state, nvt, sim_mode)

    # thermalize the thermostat
    sim.run(0)
    nvt.thermalize_thermostat_dof()

    # run
    sim.run(RUN_STEPS)

    job.document['nvt_md_complete'] = True

@LJFluid.operation(directives=dict(
    walltime=48, executable=CONFIG["executable"], nranks=8))
@LJFluid.pre.after(create_initial_state)
@LJFluid.post.true('npt_md_complete')
def run_npt_md_sim(job):
    """Run an NPT simulation at the pressure computed by the NVT simulation."""
    import hoomd
    from hoomd import md
    from custom_actions import ComputeDensity

    device = hoomd.device.CPU()
    initial_state = job.fn('initial_state.gsd')
    p = job.statepoint.pressure
    npt = md.methods.NPT(hoomd.filter.All(),
                         kT=job.sp.kT,
                         tau=0.24,
                         S=[p, p, p, 0, 0, 0],
                         tauS=2.4,
                         couple='xyz')
    sim_mode = 'npt_md'
    density_compute = ComputeDensity()

    sim = make_md_simulation(job,
                             device,
                             initial_state,
                             npt,
                             sim_mode,
                             extra_loggables=[density_compute])

    # thermalize the thermostat and barostat
    sim.run(0)
    npt.thermalize_thermostat_and_barostat_dof()

    # run
    sim.run(RUN_STEPS)

    job.document['npt_md_complete'] = True


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
    mc = hpmc.integrate.Sphere()
    mc.shape[('A', 'A')] = dict(diameter=0.0)

    # pair potential
    epsilon = LJ_PARAMS['epsilon'] / job.sp.kT  # noqa F841
    sigma = LJ_PARAMS['sigma']
    r_on = LJ_PARAMS['r_on']
    r_cut = LJ_PARAMS['r_cut']

    # the potential will have xplor smoothing with r_on=2
    lj_str = """// standard lj energy with sigma set to 1
                float rsq = dot(r_ij, r_ij);
                float sigma = {sigma:.15f};
                float sigsq = sigma * sigma;
                float rsqinv = sigsq / rsq;
                float r6inv = rsqinv * rsqinv * rsqinv;
                float r12inv = r6inv * r6inv;
                float energy = 4 * {epsilon:.15f} * (r12inv - r6inv);

                // apply xplor smoothing
                float r_on = {r_on:.15f};
                float r_cut = {r_cut:.15f};
                float r = sqrtf(rsq);
                if (r > r_on && r <= r_cut)
                {{
                    // computing denominator for the shifting factor
                    float r_onsq = r_on * r_on;
                    float r_cutsq = r_cut * r_cut;
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

    patch = hpmc.pair.user.CPPPotential(r_cut=r_cut,
                                        code=lj_str,
                                        param_array=[])
    mc.pair_potential = patch

    # log to gsd
    logger_gsd = hoomd.logging.Logger(categories=['scalar', 'sequence'])
    logger_gsd.add(patch, quantities=['energy'])
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
    mstuner = hpmc.tune.MoveSize.scale_solver(moves=['d'],
                                              target=0.2,
                                              trigger=hoomd.trigger.And([
                                                  hoomd.trigger.Periodic(10),
                                                  hoomd.trigger.Before(10000)
                                              ]))
    sim.operations.add(mstuner)

    return sim


@LJFluid.operation(directives=dict(
    walltime=48, executable=CONFIG["executable"], nranks=8))
@LJFluid.pre.after(create_initial_state)
@LJFluid.post.true('nvt_mc_complete')
def run_nvt_mc_sim(job):
    """Run MC sim in NVT."""
    import hoomd

    # simulation
    dev = hoomd.device.CPU()
    initial_state = job.fn('initial_state.gsd')
    sim_mode = 'nvt_mc'
    sim = make_mc_simulation(job, dev, initial_state, sim_mode)

    # run
    sim.run(RUN_STEPS)

    job.document['nvt_mc_complete'] = True

@LJFluid.operation(directives=dict(
    walltime=48, executable=CONFIG["executable"], nranks=8))
@LJFluid.pre.after(create_initial_state)
@LJFluid.post.true('npt_mc_complete')
def run_npt_mc_sim(job):
    """Run MC sim in NPT."""
    import hoomd
    from hoomd import hpmc
    from custom_actions import ComputeDensity

    # device
    dev = hoomd.device.CPU()
    initial_state = job.fn('initial_state.gsd')
    sim_mode = 'npt_mc'

    # compute the density
    compute_density = ComputeDensity()

    # box updates
    boxmc = hpmc.update.BoxMC(betaP=job.statepoint.pressure
                              / job.sp.kT,
                              trigger=hoomd.trigger.Periodic(1))
    boxmc.volume = dict(weight=1.0, mode='ln', delta=0.001)

    # simulation
    sim = make_mc_simulation(job,
                             dev,
                             initial_state,
                             sim_mode,
                             extra_loggables=[compute_density, boxmc])

    sim.operations.add(boxmc)

    boxmc_tuner = hpmc.tune.BoxMCMoveSize.scale_solver(
        trigger=hoomd.trigger.And(
            [hoomd.trigger.Periodic(200),
             hoomd.trigger.Before(100000)]),
        boxmc=boxmc,
        moves=['volume'],
        target=0.5)
    sim.operations.add(boxmc_tuner)

    # run
    sim.run(RUN_STEPS)

    job.document['npt_mc_complete'] = True

# def all_sims_analyzed(*jobs):
#     """Make sure all sims have values for potential energy computed.

#     Implicit here is the nvt_md sim, whose analysis is a precondition for
#     running the other modes of simulation.
#     """
#     for job in jobs:
#         if job.doc.npt_mc.potential_energy is None or \
#             job.doc.npt_md.potential_energy is None or \
#                 job.doc.nvt_mc.potential_energy is None or \
#                   job.doc.langevin_md.potential_energy is None:
#             return False
#     return True


# @aggregator.groupby(['kT', 'density'])
# @LJFluid.operation.with_directives(
#     directives=dict(executable=CONFIG["executable"]))
# @LJFluid.pre(all_sims_analyzed)
# def analyze_potential_energies(*jobs):
#     """Plot standard error of the mean of the potential energies.

#     This will average the potential energy value reported in the replicates.
#     """
#     import matplotlib.pyplot as plt

#     # grab the common statepoint parameters
#     kT = jobs[0].sp.kT
#     density = jobs[0].sp.density
#     num_particles = jobs[0].sp.num_particles

#     simulation_modes = ['langevin_md', 'nvt_md', 'npt_md', 'nvt_mc', 'npt_mc']

#     # organize data from jobs
#     energies = {mode: [] for mode in simulation_modes}
#     for jb in jobs:
#         for sim_mode in simulation_modes:
#             energies[sim_mode].append(
#                 getattr(getattr(jb.doc, sim_mode), 'potential_energy'))

#     # compute stats with data
#     avg_energy_pp = {
#         sim_mode: np.mean(energies[sim_mode]) / num_particles
#         for sim_mode in simulation_modes
#     }
#     stderr_energy_pp = {
#         sim_mode: 2 * np.std(energies[sim_mode])
#         / np.sqrt(len(energies[sim_mode])) / num_particles
#         for sim_mode in simulation_modes
#     }

#     # compute the energy differences
#     egy_pp_list = [avg_energy_pp[mode] for mode in simulation_modes]
#     stderr_pp_list = [stderr_energy_pp[mode] for mode in simulation_modes]
#     avg_across_modes = np.mean(egy_pp_list)
#     egy_diff_list = np.array(egy_pp_list) - avg_across_modes

#     # make plot
#     plt.bar(range(len(simulation_modes)),
#             height=egy_diff_list,
#             yerr=stderr_pp_list,
#             alpha=0.5,
#             ecolor='black',
#             capsize=10)
#     plt.xticks(range(len(simulation_modes)), simulation_modes)
#     plt.title(f"$kT={kT}$, $\\rho={density}$, $N={num_particles}$")
#     plt.ylabel("$(U - <U>) / N$")
#     plt.savefig(f'potential_energies_kT{kT}_density{density}.png',
#                 bbox_inches='tight')
#     plt.show()
#     plt.close()


if __name__ == "__main__":
    LJFluid.get_project(test_project_dict["LJFluid"].path).main()
