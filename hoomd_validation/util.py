# Copyright (c) 2022-2023 The Regents of the University of Michigan.
# Part of HOOMD-blue, released under the BSD 3-Clause License.

"""Helper functions for grabbing data and plotting."""

import numpy
import io
import gsd.fl
import signac


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

        for writer in sim.operations.writers:
            if hasattr(writer, 'flush'):
                writer.flush()

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
    sim.seed = make_seed(job, sim_mode)
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
        logger=logger)
    sim.operations.add(quantity_writer)

    return sim


def make_seed(job, sim_mode=None):
    """Make a random number seed from a job.

    Mix in the simulation mode to ensure that separate simulations in the same
    state point run with different seeds.
    """
    # copy the job statepoint and mix in the simulation mode
    statepoint = job.statepoint()
    statepoint['sim_mode'] = sim_mode

    return int(signac.job.calc_id(statepoint), 16) & 0xffff


def plot_distribution(ax, data, xlabel, expected=None, bins=100):
    """Plot distributions."""
    import numpy

    max_density_histogram = 0
    sim_modes = data.keys()

    range_min = min(min(x) for x in data.values())
    range_max = max(max(x) for x in data.values())

    for mode in sim_modes:
        data_arr = numpy.asarray(data[mode])
        histogram, bin_edges = numpy.histogram(data_arr,
                                               bins=bins,
                                               range=(range_min, range_max),
                                               density=True)

        if numpy.all(data_arr == data_arr[0]):
            histogram[:] = 0

        max_density_histogram = max(max_density_histogram, numpy.max(histogram))

        ax.plot(bin_edges[:-1], histogram, label=mode)

    ax.set_xlabel(xlabel)
    ax.set_ylabel('probability density')

    if callable(expected):
        ax.plot(bin_edges[:-1],
                expected(bin_edges[:-1]),
                linestyle='dashed',
                color='k',
                label='expected')

    elif expected is not None:
        ax.vlines(x=expected,
                  ymin=0,
                  ymax=max_density_histogram,
                  linestyles='dashed',
                  colors='k')


def plot_vs_expected(ax,
                     values,
                     ylabel,
                     expected=0,
                     relative_scale=None,
                     separate_nvt_npt=False):
    """Plot values vs an expected value."""
    sim_modes = values.keys()

    avg_value = {mode: numpy.mean(values[mode]) for mode in sim_modes}
    stderr_value = {
        mode: 2 * numpy.std(values[mode]) / numpy.sqrt(len(values[mode]))
        for mode in sim_modes
    }

    # compute the energy differences
    value_list = [avg_value[mode] for mode in sim_modes]
    stderr_list = numpy.array([stderr_value[mode] for mode in sim_modes])

    value_diff_list = numpy.array(value_list)

    if relative_scale is not None:
        value_diff_list = (value_diff_list
                           - expected) / expected * relative_scale
        stderr_list = stderr_list / expected * relative_scale

    ax.errorbar(x=range(len(sim_modes)),
                y=value_diff_list,
                yerr=numpy.fabs(stderr_list),
                fmt='s')
    ax.set_xticks(range(len(sim_modes)), sim_modes, rotation=45)
    ax.set_ylabel(ylabel)

    if separate_nvt_npt:
        # Indicate average nvt and npt values separately.
        npt_modes = list(filter(lambda x: 'npt' in x, sim_modes))
        npt_mean = numpy.mean([avg_value[mode] for mode in npt_modes])
        nvt_modes = list(filter(lambda x: 'nvt' in x, sim_modes))
        nvt_mean = numpy.mean([avg_value[mode] for mode in nvt_modes])

        if relative_scale is not None:
            npt_mean = (npt_mean - expected) / expected * relative_scale
            nvt_mean = (nvt_mean - expected) / expected * relative_scale

        # _sort_sim_modes places npt modes first
        ax.hlines(y=npt_mean,
                  xmin=0,
                  xmax=len(npt_modes) - 1,
                  linestyles='dashed',
                  colors='k')

        ax.hlines(y=nvt_mean,
                  xmin=len(npt_modes),
                  xmax=len(sim_modes) - 1,
                  linestyles='dashed',
                  colors='k')
    else:
        ax.hlines(y=expected,
                  xmin=0,
                  xmax=len(sim_modes) - 1,
                  linestyles='dashed',
                  colors='k')

    return avg_value, stderr_value


def plot_timeseries(ax,
                    timesteps,
                    data,
                    ylabel,
                    expected=None,
                    max_points=None):
    """Plot data as a time series."""
    provided_modes = list(data.keys())

    for mode in provided_modes:
        if max_points is not None and len(data[mode]) > max_points:
            skip = len(data[mode]) // max_points
            plot_data = numpy.asarray(data[mode][::skip])
            plot_timestep = numpy.asarray(timesteps[mode][::skip])
        else:
            plot_data = numpy.asarray(data[mode])
            plot_timestep = numpy.asarray(timesteps[mode])

        ax.plot(plot_timestep, plot_data, label=mode)

    ax.set_xlabel("time step")
    ax.set_ylabel(ylabel)

    if expected is not None:
        ax.hlines(y=expected,
                  xmin=0,
                  xmax=timesteps[provided_modes[0]][-1],
                  linestyles='dashed',
                  colors='k')
