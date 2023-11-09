# Copyright (c) 2022-2023 The Regents of the University of Michigan.
# Part of HOOMD-blue, released under the BSD 3-Clause License.

"""Helper functions for grabbing data and plotting."""

import os

import h5py
import numpy
import signac


def true_all(*jobs, key):
    """Check that a given key is true in all jobs."""
    return all(job.document.get(key, False) for job in jobs)


def total_ranks_function(ranks_per_job):
    """Make a function that computes the number of ranks for an aggregate."""
    return lambda *jobs: ranks_per_job * len(jobs)


def get_job_filename(sim_mode, device, name, file_type):
    """Construct a job filename."""
    import hoomd

    suffix = 'cpu'
    if isinstance(device, hoomd.device.GPU):
        suffix = 'gpu'

    return f'{sim_mode}_{suffix}_{name}.{file_type}'


def get_message_filename(job, filename):
    """Get a cluster job unique message filename."""
    cluster_id = os.environ.get('SLURM_JOB_ID', None)
    if cluster_id is not None:
        return job.fn(f'{cluster_id}-{filename}')

    return job.fn(filename)


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
        if next_walltime >= walltime_stop:
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
    trajectory_logger=None,
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

        trajectory_logger (`hoomd.logging.Logger`): Logger to add to trajectory
            writer.
    """
    import hoomd

    sim = hoomd.Simulation(device)
    sim.seed = make_seed(job, sim_mode)
    sim.create_state_from_gsd(initial_state)

    sim.operations.integrator = integrator

    # write to notice file
    file = hoomd.device.NoticeFile(device)
    logger_table = hoomd.logging.Logger(categories=['scalar'])
    logger_table.add(sim, quantities=['timestep', 'final_timestep', 'tps'])
    logger_table[('walltime')] = (sim.device.communicator, 'walltime', 'scalar')
    table_writer = hoomd.write.Table(
        hoomd.trigger.Periodic(table_write_period), logger_table, output=file
    )
    sim.operations.add(table_writer)

    # write particle trajectory to a gsd file
    trajectory_writer = hoomd.write.GSD(
        filename=job.fn(get_job_filename(sim_mode, device, 'trajectory', 'gsd')),
        trigger=hoomd.trigger.And(
            [
                hoomd.trigger.Periodic(trajectory_write_period),
                hoomd.trigger.After(log_start_step),
            ]
        ),
        logger=trajectory_logger,
        mode='ab',
    )
    sim.operations.add(trajectory_writer)

    # write logged quantities to h5 file
    logger.add(sim, quantities=['timestep'])

    quantity_writer = hoomd.write.HDF5Log(
        filename=job.fn(get_job_filename(sim_mode, device, 'quantities', 'h5')),
        trigger=hoomd.trigger.And(
            [
                hoomd.trigger.Periodic(log_write_period),
                hoomd.trigger.After(log_start_step),
            ]
        ),
        mode='w',
        logger=logger,
    )
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

    return int(signac.job.calc_id(statepoint), 16) & 0xFFFF


def plot_distribution(
    ax, data, independent_variable_label, expected=None, bins=100, plot_rotated=False
):
    """Plot distributions."""
    import numpy

    max_density_histogram = 0
    sim_modes = data.keys()

    range_min = min(min(x) for x in data.values())
    range_max = max(max(x) for x in data.values())

    for mode in sim_modes:
        data_arr = numpy.asarray(data[mode])
        histogram, bin_edges = numpy.histogram(
            data_arr, bins=bins, range=(range_min, range_max), density=True
        )
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

        if numpy.all(data_arr == data_arr[0]):
            histogram[:] = 0

        max_density_histogram = max(max_density_histogram, numpy.max(histogram))

        if plot_rotated:
            ax.set_ylabel(independent_variable_label)
            ax.set_xlabel('probability density')
            ax.plot(histogram, bin_centers, label=mode)
        else:
            ax.set_xlabel(independent_variable_label)
            ax.set_ylabel('probability density')
            ax.plot(bin_centers, histogram, label=mode)

    if callable(expected):
        if plot_rotated:
            ax.plot(
                expected(bin_centers),
                bin_centers,
                linestyle='dashed',
                color='k',
                label='expected',
            )
        else:
            ax.plot(
                bin_centers,
                expected(bin_centers),
                linestyle='dashed',
                color='k',
                label='expected',
            )

    elif expected is not None:
        if plot_rotated:
            ax.hlines(
                y=expected,
                xmin=0,
                xmax=max_density_histogram,
                linestyles='dashed',
                colors='k',
            )
        else:
            ax.vlines(
                x=expected,
                ymin=0,
                ymax=max_density_histogram,
                linestyles='dashed',
                colors='k',
            )


def plot_vs_expected(
    ax, values, ylabel, expected=0, relative_scale=None, separate_nvt_npt=False
):
    """Plot values vs an expected value."""
    sim_modes = values.keys()

    avg_value = {}
    stderr_value = {}
    for mode in sim_modes:
        if numpy.all(numpy.isnan(values[mode])):
            avg_value[mode] = numpy.nan
            stderr_value[mode] = numpy.nan
        else:
            avg_value[mode] = numpy.mean(values[mode])
            stderr_value[mode] = (
                2 * numpy.std(values[mode]) / numpy.sqrt(len(values[mode]))
            )

    # compute the energy differences
    value_list = [avg_value[mode] for mode in sim_modes]
    stderr_list = numpy.array([stderr_value[mode] for mode in sim_modes])

    value_diff_list = numpy.array(value_list)

    if relative_scale is not None:
        value_diff_list = (value_diff_list - expected) / expected * relative_scale
        stderr_list = stderr_list / expected * relative_scale

    ax.errorbar(
        x=range(len(sim_modes)),
        y=value_diff_list,
        yerr=numpy.fabs(stderr_list),
        fmt='s',
    )
    ax.set_xticks(range(len(sim_modes)), sim_modes, rotation=45)
    ax.set_ylabel(ylabel)

    if separate_nvt_npt:
        # Indicate average nvt and npt values separately.
        npt_modes = list(filter(lambda x: 'npt' in x, sim_modes))
        npt_mean = numpy.nanmean([avg_value[mode] for mode in npt_modes])
        nvt_modes = list(filter(lambda x: 'nvt' in x or 'nec' in x, sim_modes))
        nvt_mean = numpy.nanmean([avg_value[mode] for mode in nvt_modes])

        if relative_scale is not None:
            npt_mean = (npt_mean - expected) / expected * relative_scale
            nvt_mean = (nvt_mean - expected) / expected * relative_scale

        # _sort_sim_modes places npt modes first
        ax.hlines(
            y=npt_mean, xmin=0, xmax=len(npt_modes) - 1, linestyles='dashed', colors='k'
        )

        ax.hlines(
            y=nvt_mean,
            xmin=len(npt_modes),
            xmax=len(sim_modes) - 1,
            linestyles='dashed',
            colors='k',
        )
    else:
        ax.hlines(
            y=expected, xmin=0, xmax=len(sim_modes) - 1, linestyles='dashed', colors='k'
        )

    return avg_value, stderr_value


def plot_timeseries(ax, timesteps, data, ylabel, expected=None, max_points=None):
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

    ax.set_xlabel('time step')
    ax.set_ylabel(ylabel)

    if expected is not None:
        ax.hlines(
            y=expected,
            xmin=0,
            xmax=timesteps[provided_modes[0]][-1],
            linestyles='dashed',
            colors='k',
        )


def _sort_sim_modes(sim_modes):
    """Sort simulation modes for comparison."""
    sim_modes.sort(key=lambda x: ('nvt' in x or 'nec' in x, 'md' in x, x))


def read_log(filename):
    """Read a HDF5 log as a dictionary of logged quantities."""
    with h5py.File(mode='r', name=filename) as f:
        keys = []
        f.visit(lambda name: keys.append(name))
        return {key: numpy.array(f[key]) for key in keys}
