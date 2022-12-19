# Copyright (c) 2022 The Regents of the University of Michigan.
# Part of HOOMD-blue, released under the BSD 3-Clause License.

"""Helper functions for grabbing data and plotting."""

import numpy
import io
import gsd.fl


def read_gsd_log_trajectory(traj):
    """Read the log values from a GSD trajectory.

    Args:
        traj (`gsd.hoomd.Trajectory`): trajectory to read data from

    Reading GSD file is expensive, call `read_gsd_log_trajectory` once, then
    call `get_log_quantity` multiple times to extract individual log quantities.
    """
    return [frame.log for frame in traj]


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


def total_ranks_function(ranks_per_job):
    """Make a function that computes the number of ranks for an aggregate."""
    return lambda *jobs: ranks_per_job * len(jobs)


def gsd_step_greater_equal_function(gsd_filename, step):
    """Make a function that compares the timestep in the gsd to step.

    Returns `True` when the final timestep in ``job.fn(gsd_filename)`` is
    greater than or equal to ``step``.

    The function returns `False` for files that do not exist and files that
    have no frames.
    """

    def gsd_step_greater_equal(*jobs):
        for job in jobs:
            if not job.isfile(gsd_filename):
                return False

            try:
                with gsd.fl.open(name=job.fn(gsd_filename), mode='rb') as f:
                    if f.nframes == 0:
                        return False

                    last_frame = f.nframes - 1

                    if f.chunk_exists(frame=last_frame,
                                      name='configuration/step'):
                        gsd_step = f.read_chunk(frame=last_frame,
                                                name='configuration/step')[0]
                        if gsd_step < step:
                            return False
                    else:
                        return False
            except RuntimeError:
                # treat corrupt GSD files as not complete, as these are still
                # being written.
                return False

        return True

    return gsd_step_greater_equal


def get_job_filename(sim_mode, device, name, type):
    """Construct a job filename."""
    import hoomd

    suffix = 'cpu'
    if isinstance(device, hoomd.device.GPU):
        suffix = 'gpu'

    return f"{sim_mode}_{suffix}_{name}.{type}"


def run_up_to_walltime(sim, end_step, steps, walltime_stop):
    """Run a simulation, stopping early if a walltime limit is reached.

    Args:
        sim (hoomd.Simulation): simulation object.
        end_step (int): Timestep to stop at.
        steps (int): Number of steps to run in each batch.
        walltime_stop (int): Walltime (in seconds) to stop at.
    """
    while sim.timestep < end_step:
        sim.run(min(steps, end_step - sim.timestep))

        next_walltime = sim.device.communicator.walltime + sim.walltime
        if (next_walltime >= walltime_stop):
            break


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
    log_start_step,
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

        log_start_step (int): Timestep to start writing trajectories.
    """
    import hoomd

    sim = hoomd.Simulation(device)
    sim.seed = job.statepoint.replicate_idx
    sim.create_state_from_gsd(initial_state)

    sim.operations.integrator = integrator

    # write to terminal
    if sim.device.communicator.rank == 0:
        file = open(job.fn(get_job_filename(sim_mode, device, 'tps', 'log')),
                    mode='w',
                    newline='\n')
    else:
        file = io.StringIO("")
    logger_table = hoomd.logging.Logger(categories=['scalar'])
    logger_table.add(sim, quantities=['timestep', 'final_timestep', 'tps'])
    logger_table[('walltime')] = (sim.device.communicator, 'walltime', 'scalar')
    table_writer = hoomd.write.Table(hoomd.trigger.Periodic(table_write_period),
                                     logger_table,
                                     output=file)
    sim.operations.add(table_writer)

    # write particle trajectory to gsd file
    trajectory_writer = hoomd.write.GSD(
        filename=job.fn(get_job_filename(sim_mode, device, 'trajectory',
                                         'gsd')),
        trigger=hoomd.trigger.And([
            hoomd.trigger.Periodic(trajectory_write_period),
            hoomd.trigger.After(log_start_step)
        ]),
        mode='ab')
    sim.operations.add(trajectory_writer)

    # write logged quantities to gsd file
    quantity_writer = hoomd.write.GSD(
        filter=hoomd.filter.Null(),
        filename=job.fn(get_job_filename(sim_mode, device, 'quantities',
                                         'gsd')),
        trigger=hoomd.trigger.And([
            hoomd.trigger.Periodic(log_write_period),
            hoomd.trigger.After(log_start_step)
        ]),
        mode='ab',
        log=logger)
    sim.operations.add(quantity_writer)

    return sim
