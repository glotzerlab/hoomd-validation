# Copyright (c) 2022 The Regents of the University of Michigan.
# Part of HOOMD-blue, released under the BSD 3-Clause License.

"""Lennard Jones phase behavior validation test."""

import numpy as np
from config import test_project_dict, CONTAINER_IMAGE_PATH
from project_classes import LJFluid
from flow import aggregator


@LJFluid.operation.with_directives(directives=dict(
    executable="singularity exec {} python".format(CONTAINER_IMAGE_PATH)))
@LJFluid.post.isfile('initial_state.gsd')
def create_initial_state(job):
    """Create initial system configuration."""
    import gsd.hoomd
    import itertools

    sp = job.sp()

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


@LJFluid.operation.with_directives(directives=dict(
    walltime=48,
    executable="singularity exec {} python".format(CONTAINER_IMAGE_PATH),
    nranks=8))
@LJFluid.pre.isfile('initial_state.gsd')
@LJFluid.pre.after(create_initial_state)
@LJFluid.post.isfile('nvt_md_sim.gsd')
def run_nvt_md_sim(job):
    """Run the MD simulation in NVT."""
    import hoomd
    from hoomd import md

    sp = job.sp()
    doc = job.doc()

    device = hoomd.device.CPU()
    sim = hoomd.Simulation(device)
    sim.create_state_from_gsd(job.fn('initial_state.gsd'))
    sim.seed = doc["nvt_md"]["seed"]

    # pair force
    nlist = md.nlist.Cell(buffer=0.3)
    lj = md.pair.LJ(default_r_cut=2.5, default_r_on=2.0, nlist=nlist)
    lj.params[('A', 'A')] = dict(sigma=1, epsilon=1)
    lj.shift_mode = 'xplor'

    # integration method
    nvt = md.methods.NVT(hoomd.filter.All(), kT=sp["kT"], tau=0.1)

    # integrator
    integrator = md.Integrator(dt=0.005, methods=[nvt], forces=[lj])
    sim.operations.integrator = integrator

    # compute pressure
    thermo = md.compute.ThermodynamicQuantities(hoomd.filter.All())
    sim.operations.add(thermo)

    # log the pressure
    logger = hoomd.logging.Logger(categories=['scalar'])
    logger.add(thermo, quantities=['pressure', 'potential_energy'])

    # write data to gsd file
    gsd_writer = hoomd.write.GSD(filename=job.fn('nvt_md_sim.gsd'),
                                 trigger=hoomd.trigger.Periodic(1000),
                                 mode='wb',
                                 log=logger)
    sim.operations.add(gsd_writer)

    # write to terminal
    logger_table = hoomd.logging.Logger(categories=['scalar'])
    logger_table.add(sim, quantities=['timestep', 'final_timestep', 'tps'])
    table_writer = hoomd.write.Table(hoomd.trigger.Periodic(1000), logger_table)
    sim.operations.add(table_writer)

    # thermoalize momenta
    sim.state.thermalize_particle_momenta(hoomd.filter.All(), sp["kT"])

    # run
    sim.run(2e6)


@LJFluid.operation.with_directives(directives=dict(
    executable="singularity exec {} python".format(CONTAINER_IMAGE_PATH)))
@LJFluid.pre.isfile('nvt_md_sim.gsd')
@LJFluid.pre.after(run_nvt_md_sim)
@LJFluid.post(lambda job: job.doc.nvt_md.pressure != 0.0)
@LJFluid.post(lambda job: job.doc.nvt_md.potential_energy != 0.0)
@LJFluid.post.isfile('nvt_md_pressure_vs_time.png')
def analyze_nvt_md_sim(job):
    """Compute the pressure for use in NPT simulations to cross-validate."""
    import gsd.hoomd
    import matplotlib.pyplot as plt

    traj = gsd.hoomd.open(job.fn('nvt_md_sim.gsd'))

    # analyze over the last 1000 frame (1e6 timesteps)
    traj = traj[-1000:]

    # create array of data points
    pressures = np.zeros(len(traj))
    potential_energies = np.zeros(len(traj))
    for i, frame in enumerate(traj):
        pressures[i] = frame.log['md/compute/ThermodynamicQuantities/pressure']
        potential_energies[i] = frame.log[
            'md/compute/ThermodynamicQuantities/potential_energy']

    # save the average value in a job doc parameter
    job.doc.nvt_md.pressure = np.average(pressures)
    job.doc.nvt_md.potential_energy = np.average(potential_energies)

    # make plots for visual inspection
    plt.plot(pressures)
    plt.title('Pressure vs. time')
    plt.ylabel('$P$')
    plt.savefig(job.fn('nvt_md_pressure_vs_time.png'), bbox_inches='tight')
    plt.close()

    plt.plot(potential_energies)
    plt.title('Potential Energy vs. time')
    plt.ylabel('$U$')
    plt.savefig(job.fn('nvt_md_potential_energy_vs_time.png'),
                bbox_inches='tight')
    plt.close()


@LJFluid.operation.with_directives(directives=dict(
    walltime=48,
    executable="singularity exec {} python".format(CONTAINER_IMAGE_PATH),
    nranks=8))
@LJFluid.pre.isfile('initial_state.gsd')
@LJFluid.pre(lambda job: job.doc.nvt_md.pressure != 0.0)
@LJFluid.pre.after(create_initial_state)
@LJFluid.post.isfile('npt_md_sim.gsd')
def run_npt_md_sim(job):
    """Run an npt simulation at the pressure computed by the NVT simulation."""
    import hoomd
    from hoomd import md
    from custom_actions import ComputeDensity

    sp = job.sp()
    doc = job.doc()

    device = hoomd.device.CPU()
    sim = hoomd.Simulation(device)
    sim.create_state_from_gsd(job.fn('initial_state.gsd'))
    sim.seed = doc["npt_md"]["seed"]

    # pair force
    nlist = md.nlist.Cell(buffer=0.2)
    lj = md.pair.LJ(default_r_cut=2.5, nlist=nlist)
    lj.params[('A', 'A')] = dict(sigma=1, epsilon=1)
    lj.shift_mode = 'xplor'

    # integration method
    p = doc["nvt_md"]["pressure"]
    npt = md.methods.NPT(hoomd.filter.All(),
                         kT=sp["kT"],
                         tau=0.1,
                         S=[p, p, p, 0, 0, 0],
                         tauS=0.1,
                         couple='xyz')

    # integrator
    integrator = md.Integrator(dt=0.005, methods=[npt], forces=[lj])
    sim.operations.integrator = integrator

    # compute pressure
    thermo = md.compute.ThermodynamicQuantities(hoomd.filter.All())
    sim.operations.add(thermo)

    # log quantities
    density_compute = ComputeDensity(sim, sp["num_particles"])
    logger = hoomd.logging.Logger()
    logger.add(thermo, quantities=['pressure', 'potential_energy'])
    logger.add(density_compute, quantities=['density'])

    # write data to gsd file
    gsd_writer = hoomd.write.GSD(filename=job.fn('npt_md_sim.gsd'),
                                 trigger=hoomd.trigger.Periodic(1000),
                                 mode='wb',
                                 log=logger)
    sim.operations.add(gsd_writer)

    # write to terminal
    logger_table = hoomd.logging.Logger(categories=['scalar'])
    logger_table.add(sim, quantities=['timestep', 'final_timestep', 'tps'])
    table_writer = hoomd.write.Table(hoomd.trigger.Periodic(1000), logger_table)
    sim.operations.add(table_writer)

    # thermoalize momenta
    sim.state.thermalize_particle_momenta(hoomd.filter.All(), sp["kT"])

    # run
    sim.run(2e6)


@LJFluid.operation.with_directives(directives=dict(
    executable="singularity exec {} python".format(CONTAINER_IMAGE_PATH)))
@LJFluid.pre.isfile('npt_md_sim.gsd')
@LJFluid.pre.after(run_npt_md_sim)
@LJFluid.post(lambda job: job.doc.npt_md.potential_energy != 0.0)
def analyze_npt_md_sim(job):
    """Compute the density to cross-validate with earlier NVT simulations."""
    import gsd.hoomd
    import matplotlib.pyplot as plt

    traj = gsd.hoomd.open(job.fn('npt_md_sim.gsd'))

    # analyze over the last 1000 frame (1e6 timesteps)
    traj = traj[-1000:]

    # create array of pressures
    pressures = np.zeros(len(traj))
    potential_energies = np.zeros(len(traj))
    densities = np.zeros(len(traj))
    for i, frame in enumerate(traj):
        pressures[i] = frame.log['md/compute/ThermodynamicQuantities/pressure']
        potential_energies[i] = frame.log[
            'md/compute/ThermodynamicQuantities/potential_energy']
        densities[i] = frame.log['custom_actions/ComputeDensity/density']

    # save the average value in a job doc parameter
    job.doc.npt_md.density = np.average(densities)
    job.doc.npt_md.potential_energy = np.average(potential_energies)

    # make plots for visual inspection
    plt.plot(pressures)
    plt.title('Pressure vs. time')
    plt.ylabel('$P \\sigma^3 / \\epsilon$')
    plt.savefig(job.fn('npt_md_pressure_vs_time.png'), bbox_inches='tight')
    plt.close()

    plt.plot(potential_energies)
    plt.title('Potential Energy vs. time')
    plt.ylabel('$U / \\epsilon$')
    plt.savefig(job.fn('npt_md_potential_energy_vs_time.png'),
                bbox_inches='tight')
    plt.close()

    plt.plot(densities)
    plt.title('Number Density vs. time')
    plt.ylabel('$\\rho \\sigma^3$')
    plt.savefig(job.fn('npt_md_density_vs_time.png'), bbox_inches='tight')
    plt.close()


@LJFluid.operation.with_directives(directives=dict(
    walltime=48,
    executable="singularity exec {} python".format(CONTAINER_IMAGE_PATH),
    nranks=16))
@LJFluid.pre.isfile('initial_state.gsd')
@LJFluid.pre.after(create_initial_state)
@LJFluid.post.isfile('nvt_mc_sim.gsd')
def run_nvt_mc_sim(job):
    """Run MC sim in NVT."""
    import hoomd
    from hoomd import hpmc

    sp = job.sp()
    doc = job.doc()

    # simulation
    dev = hoomd.device.CPU()
    sim = hoomd.Simulation(dev)
    sim.create_state_from_gsd(job.fn('initial_state.gsd'))
    sim.seed = doc["nvt_mc"]["seed"]

    # integrator
    mc = hpmc.integrate.Sphere()
    mc.shape[('A', 'A')] = dict(diameter=0.0)
    sim.operations.integrator = mc

    # pair potential
    epsilon = 1 / sp["kT"]  # noqa F841
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

    gsd_writer = hoomd.write.GSD(filename=job.fn('nvt_mc_sim.gsd'),
                                 trigger=hoomd.trigger.Periodic(1000),
                                 mode='wb',
                                 log=logger_gsd)
    sim.operations.add(gsd_writer)

    # write to terminal
    logger_table = hoomd.logging.Logger(categories=['scalar'])
    logger_table.add(sim, quantities=['timestep', 'final_timestep', 'tps'])
    table_writer = hoomd.write.Table(hoomd.trigger.Periodic(1000), logger_table)
    sim.operations.add(table_writer)

    # make sure we have a valid initial state
    sim.run(0)
    if mc.overlaps > 0:
        raise RuntimeError("Initial configuration has overlaps!")

    # run
    sim.run(2e6)


@LJFluid.operation.with_directives(directives=dict(
    executable="singularity exec {} python".format(CONTAINER_IMAGE_PATH)))
@LJFluid.pre.isfile('nvt_mc_sim.gsd')
@LJFluid.pre.after(run_nvt_mc_sim)
@LJFluid.post(lambda job: job.doc.nvt_mc.potential_energy != 0.0)
def analyze_nvt_mc_sim(job):
    """Compute the pressure for use in NPT simulations to cross-validate."""
    import gsd.hoomd
    import matplotlib.pyplot as plt

    sp = job.sp()

    traj = gsd.hoomd.open(job.fn('nvt_mc_sim.gsd'))

    # analyze over the last 1000 frame (1e6 timesteps)
    traj = traj[-1000:]

    # create array of data points
    potential_energies = np.zeros(len(traj))

    for i, frame in enumerate(traj):
        potential_energies[i] = frame.log['hpmc/pair/user/CPPPotential/energy']

    # need to scale this energy by kT, since the simulation uses epsilon=1/kT,
    # so we can compare back to the energies of the MD simulations
    job.doc.nvt_mc.potential_energy = sp["kT"] * np.average(potential_energies)

    # make plots for visual inspection
    plt.plot(potential_energies)
    plt.title('Potential Energy vs. time')
    plt.ylabel('$U$')
    plt.savefig(job.fn('nvt_mc_potential_energy_vs_time.png'),
                bbox_inches='tight')
    plt.close()


@LJFluid.operation.with_directives(directives=dict(
    walltime=48,
    executable="singularity exec {} python".format(CONTAINER_IMAGE_PATH),
    nranks=16))
@LJFluid.pre.isfile('initial_state.gsd')
@LJFluid.pre.after(run_nvt_md_sim)
@LJFluid.pre(lambda job: job.doc.nvt_md.pressure > 0.0)
@LJFluid.post.isfile('npt_mc_sim.gsd')
def run_npt_mc_sim(job):
    """Run MC sim in NPT."""
    import hoomd
    from hoomd import hpmc
    from custom_actions import ComputeDensity

    sp = job.sp()
    doc = job.doc()

    # simulation
    dev = hoomd.device.CPU()
    sim = hoomd.Simulation(dev)
    sim.create_state_from_gsd(job.fn('initial_state.gsd'))
    sim.seed = doc["npt_mc"]["seed"]

    # integrator
    mc = hpmc.integrate.Sphere()
    mc.shape[('A', 'A')] = dict(diameter=0.0)
    sim.operations.integrator = mc

    # pair potential
    epsilon = 1 / sp["kT"]  # noqa F841
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

    patch = hpmc.pair.user.CPPPotential(r_cut=2.5, code=lj_str, param_array=[])
    mc.pair_potential = patch

    # update box
    boxmc = hpmc.update.BoxMC(betaP=doc["nvt_md"]["pressure"] / sp["kT"],
                              trigger=hoomd.trigger.Periodic(10))
    boxmc.volume = dict(weight=1.0, mode='standard', delta=25)
    sim.operations.add(boxmc)

    trigger_tuners = hoomd.trigger.And(
        [hoomd.trigger.Periodic(10),
         hoomd.trigger.Before(10000)])

    # tune box updates
    mstuner_boxmc = hpmc.tune.BoxMCMoveSize(boxmc=boxmc,
                                            trigger=trigger_tuners,
                                            moves=['volume'],
                                            target=0.2,
                                            solver=hoomd.tune.ScaleSolver(),
                                            max_move_size=dict(volume=50.0))
    sim.operations.add(mstuner_boxmc)

    # move size tuner
    mstuner = hpmc.tune.MoveSize.scale_solver(moves=['d'],
                                              target=0.2,
                                              trigger=trigger_tuners)
    sim.operations.add(mstuner)

    # log to gsd
    compute_density = ComputeDensity(sim, sp["num_particles"])
    logger_gsd = hoomd.logging.Logger()
    logger_gsd.add(mc, quantities=['type_shapes'])
    logger_gsd.add(patch, quantities=['energy'])
    logger_gsd.add(compute_density, quantities=['density'])

    gsd_writer = hoomd.write.GSD(filename=job.fn('npt_mc_sim.gsd'),
                                 trigger=hoomd.trigger.Periodic(1000),
                                 mode='wb',
                                 log=logger_gsd)
    sim.operations.add(gsd_writer)

    # write to terminal
    logger_table = hoomd.logging.Logger(categories=['scalar'])
    logger_table.add(sim, quantities=['timestep', 'final_timestep', 'tps'])
    table_writer = hoomd.write.Table(hoomd.trigger.Periodic(1000), logger_table)
    sim.operations.add(table_writer)

    # make sure we have a valid initial state
    sim.run(0)
    if mc.overlaps > 0:
        raise RuntimeError("Initial configuration has overlaps!")

    # run
    sim.run(2e6)


@LJFluid.operation.with_directives(directives=dict(
    executable="singularity exec {} python".format(CONTAINER_IMAGE_PATH)))
@LJFluid.pre.isfile('npt_mc_sim.gsd')
@LJFluid.pre.after(run_npt_mc_sim)
@LJFluid.post(lambda job: job.doc.npt_mc.potential_energy != 0.0)
def analyze_npt_mc_sim(job):
    """Compute the density to cross-validate with earlier NVT simulations."""
    import gsd.hoomd
    import matplotlib.pyplot as plt

    sp = job.sp()

    traj = gsd.hoomd.open(job.fn('npt_mc_sim.gsd'))

    # analyze over the last 1000 frame (1e6 timesteps)
    traj = traj[-1000:]

    # create array of pressures
    potential_energies = np.zeros(len(traj))
    densities = np.zeros(len(traj))

    for i, frame in enumerate(traj):
        potential_energies[i] = frame.log['hpmc/pair/user/CPPPotential/energy']
        densities[i] = frame.log['custom_actions/ComputeDensity/density']

    # save the average value in a job doc parameter
    job.doc.npt_mc.density = np.average(densities)

    # need to scale this energy by kT, since the simulation uses epsilon=1/kT,
    # so we can compare back to the energies of the MD simulations
    job.doc.npt_mc.potential_energy = sp["kT"] * np.average(potential_energies)

    plt.plot(potential_energies)
    plt.title('Potential Energy vs. time')
    plt.ylabel('$U / \\epsilon$')
    plt.savefig(job.fn('npt_mc_potential_energy_vs_time.png'),
                bbox_inches='tight')
    plt.close()

    plt.plot(densities)
    plt.title('Number Density vs. time')
    plt.ylabel('$\\rho \\sigma^3$')
    plt.savefig(job.fn('npt_mc_density_vs_time.png'), bbox_inches='tight')
    plt.close()


# @aggregator.groupby(['kT', 'density'])
@aggregator(select=lambda job: job.sp.kT == 1.5 and job.sp.density == 0.6
            )  # eventually move this to a groupby
@LJFluid.operation
#@LJFluid.operation.with_directives(directives=dict(
#    executable="singularity exec {} python".format(CONTAINER_IMAGE_PATH)))
@LJFluid.pre.after(analyze_nvt_md_sim)
@LJFluid.pre.after(analyze_nvt_mc_sim)
@LJFluid.pre.after(analyze_npt_md_sim)
@LJFluid.pre.after(analyze_npt_mc_sim)
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
        sim_mode: np.average(np.array(energies[sim_mode])) / num_particles
        for sim_mode in simulation_modes
    }
    stderr_energy_pp = {
        sim_mode:
        2 * np.std(energies[sim_mode]) / np.sqrt(len(energies[sim_mode])) / num_particles
        for sim_mode in simulation_modes
    }

    # compute the energy differences
    egy_pp_list = [avg_energy_pp[mode] for mode in simulation_modes]
    stderr_pp_list = [stderr_energy_pp[mode] for mode in simulation_modes]
    avg_across_modes = np.average(egy_pp_list)
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
    plt.ylabel("$(U - <U>) \ / \ N$")
    plt.savefig(f'potential_energies_kT{kT}_density{density}.png',
                bbox_inches='tight')
    plt.show()
    plt.close()


if __name__ == "__main__":
    LJFluid.get_project(test_project_dict["LJFluid"].root_directory()).main()
