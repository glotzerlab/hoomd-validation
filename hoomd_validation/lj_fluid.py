# Copyright (c) 2022 The Regents of the University of Michigan.
# Part of HOOMD-blue, released under the BSD 3-Clause License.

"""Lennard Jones phase behavior validation test.


"""

from config import test_project_dict
from project_classes import LJFluid


@LJFluid.operation
@LJFluid.post.isfile('initial_state.gsd')
def create_initial_state(job):
    """Create initial system configuration."""
    import gsd.hoomd
    import itertools

    density = ???
    num_particles = ???

    box_volume = num_particles / density
    L = box_volume**(1/3.)

    N = int(np.ceil(num_particles ** (1./3.)))
    x = np.linspace(-L / 2, L / 2, N, endpoint=False)
    position = list(itertools.product(x, repeat=3))[:num_particles]

    # create snapshot
    snap = gsd.hoomd.Snapshot()
    snap.particles.N = num_particles
    snap.particles.position = position
    snap.particles.typeid = [0] * num_particles
    snap.particles.types = ['A']

    snap.configuration.box = [L, L, L, 0, 0, 0]

    with gsd.hoomd.open(job.fn("initial_state.gsd"), "wb") as traj:
        traj.append(snap)


@LJFluid.operation
@LJFluid.pre.isfile('initial_state.gsd')
@LJFluid.pre.after(create_initial_state)
@LJFluid.post.isfile('nvt_sim.gsd')
def run_nvt_simulation(job):
    """Run the simulation in NVT."""
    import hoomd
    from hoomd import md

    epsilon = ???
    sigma = ???
    r_cut_coeff = ???
    shift_mode = ???
    seed = ???
    nlist_params = ???
    kT = ???
    tau = ???
    run_steps = ???  # 1.1e6 is what the lammps study did
    r_cut = sigma * r_cut_coeff

    device = hoomd.device.CPU()
    sim = hoomd.Simulation(device)
    sim.create_state_from_gsd(job.fn('initial_state.gsd'))
    sim.seed = seed

    # pair force
    nlist = md.nlist.Cell(**nlist_params)
    lj = md.pair.LJ(default_r_cut=r_cut, nlist=nlist)
    lj.params[('A', 'A')] = dict(sigma=sigma, epsilon=epsilon)
    lj.shift_mode = shift_mode

    # integration method
    nvt = md.methods.NVT(kT=kT, tau=tau)

    # integrator
    integrator = md.Integrator(dt=delta_t, methods=[nvt], forces=[lj])
    sim.operations.integrator = integrator

    # compute pressure
    thermo = md.thermo.ThermodynamicQuantities()
    sim.operations.add(thermo)

    # log the pressure
    logger = hoomd.logging.Logger()
    logger.add(thermo, quantities=['pressure'])

    # write data to gsd file
    gsd_writer = hoomd.write.GSD(filename=job.fn('nvt_sim.gsd'),
                                 trigger=hoomd.trigger.Periodic(1000),
                                 mode='wb',
                                 log=logger)
    sim.operations.add(gsd_writer)

    # thermoalize momenta
    sim.state.thermalize_particle_momenta(hoomd.filter.All(), kT)

    # run
    sim.run(run_steps)


@LJFluid.operation
@LJFluid.pre.isfile('nvt_sim.gsd')
@LJFluid.pre.after(run_nvt_simulation)
@LJFluid.post(lambda job: job.doc.nvt_pressure != 0.0)
@LJFluid.post.isfile('nvt_pressure_vs_time.png')
def compute_nvt_pressure(job):
    """Compute the pressure for use in NPT simulations to cross-validate."""
    import gsd.hoomd
    import matplotlib.pyplot as plt

    traj = gsd.hoomd.open(job.fn('nvt_sim.gsd'))

    # TODO only use equilibrated part of trajectory
    traj = traj[:]

    # create array of pressures
    pressures = np.zeros(len(traj))
    for i, frame in enumerate(traj):
        pressures[i] = frame.log['md/compute/ThermoDynamicQuantities/pressure']

    # save the average value in a job doc parameter
    job.doc.nvt_pressure = np.average(pressures)

    # make a plot for visual inspection
    plt.plot(pressures)
    plt.title('Pressure vs. time')
    plt.ylabel('$P$')
    plt.savefig(job.fn('nvt_pressure_vs_time.png'), bbox_inches='tight')
    plt.close()


@LJFluid.operation
@LJFluid.pre.isfile('initial_state.gsd')
@LJFluid.pre.after(create_initial_state)
@LJFluid.post.isfile('npt_sim.gsd')
def run_npt_simulation(job):
    """Run an npt simulation at the pressure computed by the NVT simulation."""
    pass


@LJFluid.operation
@LJFluid.pre.isfile('npt_sim.gsd')
@LJFluid.pre.after(run_npt_simulation)
@LJFluid.post(lambda job: job.doc.npt_density != 0.0)
def compute_npt_density(job):
    """Compute the density to cross-validate with earlier NVT simulations."""
    pass


def run_mc_simulation(job):
    """There was talk of running MC simulations to further cross-validate the
    NVT/PT results. Not sure if its really necessary at this point."""
    pass


if __name__ == "__main__":
    LJFluid.get_project(test_project_dict["LJFluid"].root_directory()).main()
