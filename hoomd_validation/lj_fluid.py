# Copyright (c) 2022 The Regents of the University of Michigan.
# Part of HOOMD-blue, released under the BSD 3-Clause License.

"""Lennard Jones phase behavior validation test."""

import numpy as np
from config import test_project_dict, CONFIG
from project_classes import LJFluid
from flow import aggregator

# Run parameters shared between simulations
RUN_STEPS = 2e6
WRITE_PERIOD = 1000
LOG_PERIOD = 1000
FRAMES_ANALYZE = 1000


@LJFluid.operation.with_directives(directives=dict(executable=CONFIG["executable"]))
@LJFluid.post.isfile('initial_state.gsd')
def create_initial_state(job):
    """Create initial system configuration."""
    import gsd.hoomd
    import itertools

    sp = job.sp

    box_volume = sp["num_particles"] / sp["density"]
    L = box_volume**(1 / 3.)

    N = int(np.ceil(sp["num_particles"]**(1. / 3.)))
    x = np.linspace(-L / 2, L / 2, N, endpoint=False)
    position = list(itertools.product(x, repeat=3))[:sp["num_particles"]]

    # create snapshot
    snap = gsd.hoomd.Snapshot()
    snap.particles.N = sp["num_particles"]
    snap.particles.position = position
    snap.particles.typeid = [0] * sp["num_particles"]
    snap.particles.types = ['A']

    snap.configuration.box = [L, L, L, 0, 0, 0]

    with gsd.hoomd.open(job.fn("initial_state.gsd"), "wb") as traj:
        traj.append(snap)


def make_simulation(job, device, integrator):
    """Make a simulation.

    This operation returns a simulation with only the things needed for each
    type of simulation. This includes initial state, random seed, integrator,
    and table writer.

    Args:
        job (`signac.Job`): signac job object.
        device (`hoomd.device.Device`): hoomd device object.
        integrator (`hoomd.md.Integrator`): hoomd integrator object.
    """
    sim = hoomd.Simulation(device)
    sim.seed = job.doc.seed
    sim.create_state_from_gsd(job.fn('initial_state.gsd'))

    sim.operations.integrator = integrator

    # write to terminal
    logger_table = hoomd.logging.Logger(categories=['scalar'])
    logger_table.add(sim, quantities=['timestep', 'final_timestep', 'tps'])
    table_writer = hoomd.write.Table(hoomd.trigger.Periodic(WRITE_PERIOD),
                                     logger_table)
    sim.operations.add(table_writer)
    return sim


def make_md_simulation(job, device, method, gsd_filename, extra_loggables=[]):
    """Make an MD simulation.

    Args:
        job (`signac.job.Job`): Signac job object.
        device (`hoomd.device.Device`): hoomd device object.
        method (`hoomd.md.methods.Method`): hoomd integration method.
        gsd_filename (str): Path to the gsd file to write simulation data.
        extra_loggables (list): List of quantities to add to the gsd logger.
        ThermodynamicQuantities is added by default, any more quantities should
        be in this list.
    """
    # pair force
    nlist = md.nlist.Cell(buffer=0.3)
    lj = md.pair.LJ(default_r_cut=2.5, default_r_on=2.0, nlist=nlist)
    lj.params[('A', 'A')] = dict(sigma=1, epsilon=1)
    lj.mode = 'xplor'

    # integrator
    integrator = md.Integrator(dt=0.005, methods=[method], forces=[lj])

    # simulation
    sim = make_simulation(job, device, integrator)

    # compute thermo
    thermo = md.compute.ThermodynamicQuantities(hoomd.filter.All())
    sim.operations.add(thermo)

    # add gsd log quantities
    logger = hoomd.logging.Logger(categories=['scalar'])
    logger.add(thermo, quantities=['pressure', 'potential_energy'])
    for loggable in extra_loggables:
        # call attach explicitly so we can access sim state when computing the
        # loggable quantity
        loggable.attach(sim)
        logger.add(loggable)

    # write data to gsd file
    gsd_writer = hoomd.write.GSD(filename=gsd_filename,
                                 trigger=hoomd.trigger.Periodic(LOG_PERIOD),
                                 mode='wb',
                                 log=logger)
    sim.operations.add(gsd_writer)

    # thermalize momenta
    sim.state.thermalize_particle_momenta(hoomd.filter.All(), job.sp.kT)

    return sim


@LJFluid.operation.with_directives(directives=dict(walltime=48,
                                                   executable=CONFIG["executable"],
                                                   nranks=8))
@LJFluid.pre.after(create_initial_state)
@LJFluid.post.isfile('nvt_md_sim.gsd')
def run_nvt_md_sim(job):
    """Run the MD simulation in NVT."""
    import hoomd
    from hoomd import md

    device = hoomd.device.CPU()
    nvt = md.methods.NVT(hoomd.filter.All(), kT=job.sp.kT, tau=0.1)
    gsd_file = job.fn('nvt_md_sim.gsd')

    sim = make_md_simulation(job, device, nvt, gsd_file)

    # run
    sim.run(RUN_STEPS)


@LJFluid.operation.with_directives(directives=dict(executable=CONFIG["executable"]))
@LJFluid.pre.after(run_nvt_md_sim)
@LJFluid.post(lambda job: job.doc.nvt_md.pressure is not None)
@LJFluid.post(lambda job: job.doc.nvt_md.potential_energy is not None)
@LJFluid.post.isfile('nvt_md_pressure_vs_time.png')
def analyze_nvt_md_sim(job):
    """Compute the pressure for use in NPT simulations to cross-validate."""
    import gsd.hoomd
    from plotting import (get_energies, get_pressures, plot_energies,
                          plot_pressures)

    # get trajectory
    traj = gsd.hoomd.open(job.fn('nvt_md_sim.gsd'))
    traj = traj[-FRAMES_ANALYZE:]

    # get data
    pressures = get_pressures(traj)
    energies = get_energies(traj)

    # save the average value in a job doc parameter
    job.doc.nvt_md.pressure = np.mean(pressures)
    job.doc.nvt_md.potential_energy = np.mean(energies)

    # make plots
    plot_pressures(pressures, job.fn('nvt_md_pressure_vs_time.png'))
    plot_energies(energies, job.fn('nvt_md_potential_energy_vs_time.png'))


def nvt_md_pressures_averaged(*jobs):
    """Make sure the pressure setpoint is computed and added to job docs."""
    for job in jobs:
        if job.doc.nvt_md.aggregate_pressure is None:
            return False
    return True


def nvt_md_pressures_computed(*jobs):
    """Make sure the pressures for nvt md sims are computed."""
    for job in jobs:
        if job.doc.nvt_md.pressure is None:
            return False
    return True


@aggregator.groupby(['kT', 'density'])
@LJFluid.operation.with_directives(directives=dict(executable=CONFIG["executable"]))
@LJFluid.pre(nvt_md_pressures_computed)
@LJFluid.post(nvt_md_pressures_averaged)
def average_nvt_md_pressures(*jobs):
    """Average the pressures from all replicates at a given LJ statepoint.

    This operation is used so the pressure setpoint for all the NPT simulation
    replicates will have the exact same pressure setpoint. Conceptually, this
    operation is an MPI Allreduce, with the average operation, where each "rank"
    is a simulation replicate.
    """
    pressures = []
    for job in jobs:
        pressures.append(job.doc.nvt_md.pressure)

    avg = np.mean(pressures)
    for job in jobs:
        job.doc.nvt_md.aggregate_pressure = avg


@LJFluid.operation.with_directives(directives=dict(walltime=48,
                                                   executable=CONFIG["executable"],
                                                   nranks=8))
@LJFluid.pre(lambda job: job.doc.nvt_md.aggregate_pressure is not None)
@LJFluid.pre.after(create_initial_state)
@LJFluid.post.isfile('npt_md_sim.gsd')
def run_npt_md_sim(job):
    """Run an npt simulation at the pressure computed by the NVT simulation."""
    import hoomd
    from hoomd import md
    from custom_actions import ComputeDensity

    device = hoomd.device.CPU()
    p = job.doc.nvt_md.aggregate_pressure
    npt = md.methods.NPT(hoomd.filter.All(),
                         kT=job.sp.kT,
                         tau=0.1,
                         S=[p, p, p, 0, 0, 0],
                         tauS=0.1,
                         couple='xyz')
    gsd_file = job.fn('npt_md_sim.gsd')
    density_compute = ComputeDensity()

    sim = make_md_simulation(job,
                             device,
                             npt,
                             gsd_file,
                             extra_loggables=[density_compute])

    # run
    sim.run(RUN_STEPS)


@LJFluid.operation.with_directives(directives=dict(executable=CONFIG["executable"]))
@LJFluid.pre.after(run_npt_md_sim)
@LJFluid.post(lambda job: job.doc.npt_md.potential_energy is not None)
def analyze_npt_md_sim(job):
    """Compute the density to cross-validate with earlier NVT simulations."""
    import gsd.hoomd
    from plotting import (get_log_quantity, get_pressures, get_energies,
                          plot_pressures, plot_energies, plot_densities)

    traj = gsd.hoomd.open(job.fn('npt_md_sim.gsd'))
    traj = traj[-FRAMES_ANALYZE:]

    # get data
    pressures = get_pressures(traj)
    energies = get_energies(traj)
    densities = get_log_quantity(traj, 'custom_actions/ComputeDensity/density')

    # save the average value in a job doc parameter
    job.doc.npt_md.density = np.mean(densities)
    job.doc.npt_md.potential_energy = np.mean(energies)

    # make plots
    plot_pressures(pressures, job.fn('npt_md_pressure_vs_time.png'))
    plot_energies(energies, job.fn('npt_md_potential_energy_vs_time.png'))
    plot_densities(densities, job.fn('npt_md_density_vs_time.png'))


def make_mc_simulation(job,
                       device,
                       gsd_filename,
                       extra_loggables=[]):
    """Make an MC Simulation.

    Args:
        job (`signac.job.Job`): Signac job object.
        device (`hoomd.device.Device`): Device object.
        gsd_filename (str): Path to the gsd file to write simulation data.
        extra_loggables (list): List of extra loggables to log to gsd files.
        Patch energies and type shapes are logged by default.
    """
    import hoomd
    from hoomd import hpmc

    # integrator
    mc = hpmc.integrate.Sphere()
    mc.shape[('A', 'A')] = dict(diameter=0.0)

    # pair potential
    epsilon = 1 / job.sp.kT  # noqa F841
    r_on = 2.0
    r_cut = 2.5

    # the potential will have xplor smoothing with r_on=2
    lj_str = """// standard lj energy with sigma set to 1
                float rsq = dot(r_ij, r_ij);
                float rsqinv = 1 / rsq;
                float r6inv = rsqinv * rsqinv * rsqinv;
                float r12inv = r6inv * r6inv;
                float energy = 4 * {epsilon:.15f} * (r12inv - r6inv);

                // apply xplor smoothing
                float r_on = {r_on:.15f};
                float r_cut = {r_cut:.15f};
                float r = sqrt(rsq);
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
            """.format(epsilon=epsilon, r_on=r_on, r_cut=r_cut)

    patch = hpmc.pair.user.CPPPotential(r_cut=r_cut,
                                        code=lj_str,
                                        param_array=[])
    mc.pair_potential = patch

    # make simulation
    sim = make_simulation(job, device, mc)

    # move size tuner
    mstuner = hpmc.tune.MoveSize.scale_solver(moves=['d'],
                                              target=0.2,
                                              trigger=hoomd.trigger.And([
                                                  hoomd.trigger.Periodic(10),
                                                  hoomd.trigger.Before(10000)
                                              ]))
    sim.operations.add(mstuner)

    # log to gsd
    logger_gsd = hoomd.logging.Logger()
    logger_gsd.add(mc, quantities=['type_shapes'])
    logger_gsd.add(patch, quantities=['energy'])
    for loggable in extra_loggables:
        # call attach method explicitly so we can access simulation state when
        # computing the loggable quantity
        loggable.attach(sim)
        logger_gsd.add(loggable)

    gsd_writer = hoomd.write.GSD(filename=gsd_filename,
                                 trigger=hoomd.trigger.Periodic(LOG_PERIOD),
                                 mode='wb',
                                 log=logger_gsd)
    sim.operations.add(gsd_writer)

    # make sure we have a valid initial state
    sim.run(0)
    if mc.overlaps > 0:
        raise RuntimeError("Initial configuration has overlaps!")

    return sim


@LJFluid.operation.with_directives(directives=dict(walltime=48,
                                                   executable=CONFIG["executable"],
                                                   nranks=16))
@LJFluid.pre.after(create_initial_state)
@LJFluid.post.isfile('nvt_mc_sim.gsd')
def run_nvt_mc_sim(job):
    """Run MC sim in NVT."""
    import hoomd

    # simulation
    dev = hoomd.device.CPU()
    gsd_filename = job.fn('nvt_mc_sim.gsd')
    sim = make_mc_simulation(job, dev, gsd_filename)

    # run
    sim.run(RUN_STEPS)


@LJFluid.operation.with_directives(directives=dict(executable=CONFIG["executable"]))
@LJFluid.pre.after(run_nvt_mc_sim)
@LJFluid.post(lambda job: job.doc.nvt_mc.potential_energy is not None)
def analyze_nvt_mc_sim(job):
    """Compute the pressure for use in NPT simulations to cross-validate."""
    import gsd.hoomd
    from plotting import get_log_quantity, plot_energies

    traj = gsd.hoomd.open(job.fn('nvt_mc_sim.gsd'))
    traj = traj[-FRAMES_ANALYZE:]

    energies = get_log_quantity(traj, 'hpmc/pair/user/CPPPotential/energy')

    # need to scale this energy by kT, since the simulation uses epsilon=1/kT,
    # so we can compare back to the energies of the MD simulations
    energies *= job.sp.kT
    job.doc.nvt_mc.potential_energy = np.mean(energies)

    plot_energies(energies, job.fn('nvt_mc_potential_energy_vs_time.png'))


@LJFluid.operation.with_directives(directives=dict(walltime=48,
                                                   executable=CONFIG["executable"],
                                                   nranks=16))
@LJFluid.pre.after(create_initial_state)
@LJFluid.pre(lambda job: job.doc.nvt_md.aggregate_pressure is not None)
@LJFluid.post.isfile('npt_mc_sim.gsd')
def run_npt_mc_sim(job):
    """Run MC sim in NPT."""
    import hoomd
    from hoomd import hpmc
    from custom_actions import ComputeDensity

    # device
    dev = hoomd.device.CPU()
    gsd_filename = job.fn('npt_mc_sim.gsd')

    # compute the density
    compute_density = ComputeDensity()

    # simulation
    sim = make_mc_simulation(job,
                             dev,
                             gsd_filename,
                             extra_loggables=[compute_density])

    # box updates
    boxmc = hpmc.update.BoxMC(betaP=job.doc.nvt_md.aggregate_pressure
                              / job.sp.kT,
                              trigger=hoomd.trigger.Periodic(10))
    boxmc.volume = dict(weight=1.0, mode='standard', delta=25)
    sim.operations.add(boxmc)

    # run
    sim.run(RUN_STEPS)


@LJFluid.operation.with_directives(directives=dict(executable=CONFIG["executable"]))
@LJFluid.pre.after(run_npt_mc_sim)
@LJFluid.post(lambda job: job.doc.npt_mc.potential_energy is not None)
def analyze_npt_mc_sim(job):
    """Compute the density to cross-validate with earlier NVT simulations."""
    import gsd.hoomd
    from plotting import get_log_quantity, plot_energies, plot_densities

    traj = gsd.hoomd.open(job.fn('npt_mc_sim.gsd'))
    traj = traj[-FRAMES_ANALYZE:]

    energies = get_log_quantity(traj, 'hpmc/pair/user/CPPPotential/energy')
    densities = get_log_quantity(traj, 'custom_actions/ComputeDensity/density')

    # save the average value in a job doc parameter
    energies *= job.sp.kT
    job.doc.npt_mc.potential_energy = np.mean(energies)
    job.doc.npt_mc.density = np.mean(densities)

    # plot
    plot_energies(energies, job.fn('npt_mc_potential_energy_vs_time.png'))
    plot_densities(densities, job.fn('npt_mc_density_vs_time.png'))


def all_sims_analyzed(*jobs):
    """Make sure all sims have values for potential energy computed.

    Implicit here is the nvt_md sim, whose analysis is a precondition for
    running the other modes of simulation.
    """
    for job in jobs:
        if job.doc.npt_mc.potential_energy is None or \
            job.doc.npt_md.potential_energy is None or \
                job.doc.nvt_mc.potential_energy is None:
            return False
    return True


@aggregator.groupby(['kT', 'density'])
@LJFluid.operation.with_directives(directives=dict(executable=CONFIG["executable"]))
@LJFluid.pre(all_sims_analyzed)
def analyze_potential_energies(*jobs):
    """Plot standard error of the mean of the potential energies.

    This will average the potential energy value reported in the replicates.
    """
    import matplotlib.pyplot as plt

    # grab the common statepoint parameters
    kT = jobs[0].sp.kT
    density = jobs[0].sp.density
    num_particles = jobs[0].sp.num_particles

    simulation_modes = ['nvt_md', 'npt_md', 'nvt_mc', 'npt_mc']

    # organize data from jobs
    energies = {mode: [] for mode in simulation_modes}
    for jb in jobs:
        for sim_mode in simulation_modes:
            energies[sim_mode].append(
                getattr(getattr(jb.doc, sim_mode), 'potential_energy'))

    # compute stats with data
    avg_energy_pp = {
        sim_mode: np.mean(energies[sim_mode]) / num_particles
        for sim_mode in simulation_modes
    }
    stderr_energy_pp = {
        sim_mode: 2 * np.std(energies[sim_mode])
        / np.sqrt(len(energies[sim_mode])) / num_particles
        for sim_mode in simulation_modes
    }

    # compute the energy differences
    egy_pp_list = [avg_energy_pp[mode] for mode in simulation_modes]
    stderr_pp_list = [stderr_energy_pp[mode] for mode in simulation_modes]
    avg_across_modes = np.mean(egy_pp_list)
    egy_diff_list = np.array(egy_pp_list) - avg_across_modes

    # make plot
    plt.bar(range(len(simulation_modes)),
            height=egy_diff_list,
            yerr=stderr_pp_list,
            alpha=0.5,
            ecolor='black',
            capsize=10)
    plt.xticks(range(len(simulation_modes)), simulation_modes)
    plt.title(f"$kT={kT}$, $\\rho={density}$, $N={num_particles}$")
    plt.ylabel("$(U - <U>) / N$")
    plt.savefig(f'potential_energies_kT{kT}_density{density}.png',
                bbox_inches='tight')
    plt.show()
    plt.close()


if __name__ == "__main__":
    LJFluid.get_project(test_project_dict["LJFluid"].root_directory()).main()
