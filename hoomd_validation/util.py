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
