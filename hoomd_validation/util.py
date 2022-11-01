# Copyright (c) 2022 The Regents of the University of Michigan.
# Part of HOOMD-blue, released under the BSD 3-Clause License.

"""Helper functions for grabbing data and plotting."""

import copy
import numpy


def read_gsd_log_trajectory(traj):
    """Read the log values from a GSD trajectory.

    Args:
        traj (`gsd.hoomd.Trajectory`): trajectory to read data from

    Reading GSD file is expensive, call `read_gsd_log_trajectory` once, then
    call `get_log_quantity` multiple times to extract individual log quantities.
    """
    return [copy.copy(frame.log) for frame in traj]


def get_log_quantity(log_traj, quantity):
    """Get logged values from the return value of `read_gsd_log_trajectory`.

    Args:
        log_traj: trajectory to read data from
        quantity (str): name of the quantity to read
    """
    if len(log_traj[0][quantity]) == 1:
        qty_values = [frame[quantity][0] for frame in log_traj]
    else:
        qty_values = [frame[quantity] for frame in log_traj]

    return numpy.array(qty_values)


def true_all(*jobs, key):
    """Check that a given key is true in all jobs."""
    return all(job.document.get(key, False) for job in jobs)


def make_simulation(
    job,
    device,
    initial_state,
    integrator,
    sim_mode,
    logger,
    table_write_period,
    trajectory_write_period,
    log_write_period,
):
    """Make a simulation.

    Create a simulation, initialize its state, configure the table writer,
    trajectory writer, and quantity log file writer.

    Args:
        job (`signac.Job`): signac job object.

        device (`hoomd.device.Device`): hoomd device object.

        initial_state (str): Path to the gsd file to be used as an initial
            state.

        integrator (`hoomd.md.Integrator`): hoomd integrator object.

        sim_mode (str): String defining the simulation mode.

        logger (`hoomd.logging.Logger`): Logger object.

        table_write_period (int): Number of timesteps between table writes.

        trajectory_write_period (int): Number of timesteps between particle
            trajectory writes.

        log_write_period (int): Number of timesteps between log file writes.
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
    table_writer = hoomd.write.Table(hoomd.trigger.Periodic(table_write_period),
                                     logger_table)
    sim.operations.add(table_writer)

    # write particle trajectory to gsd file
    trajectory_writer = hoomd.write.GSD(
        filename=job.fn(f"{sim_mode}_{suffix}_trajectory.gsd"),
        trigger=hoomd.trigger.Periodic(trajectory_write_period),
        mode='wb')
    sim.operations.add(trajectory_writer)

    # write logged quantities to gsd file
    quantity_writer = hoomd.write.GSD(
        filter=hoomd.filter.Null(),
        filename=job.fn(f"{sim_mode}_{suffix}_quantities.gsd"),
        trigger=hoomd.trigger.Periodic(log_write_period),
        mode='wb',
        log=logger)
    sim.operations.add(quantity_writer)

    return sim
