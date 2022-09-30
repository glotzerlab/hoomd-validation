# Copyright (c) 2022 The Regents of the University of Michigan.
# Part of HOOMD-blue, released under the BSD 3-Clause License.

"""Helper functions for grabbing data and plotting."""

import numpy as np
import matplotlib.pyplot as plt


def get_log_quantity(traj, quantity):
    """Get logged values from a gsd trajectory.

    Args:
        traj (`gsd.hoomd.Trajectory`): trajectory to read data from
        quantity (str): name of the quantity to read
    """
    qty_values = np.zeros(len(traj))
    for i, frame in enumerate(traj):
        qty_values[i] = frame.log[quantity]
    return qty_values


def plot_quantity(data, savename, title, ylabel):
    """Plot a quanity using matplotlib.

    Args:
        data (:math:`(N, )` `np.ndarray`): One dimensional data to plot.
        savename (str): File path for saving the plot.
        title (str): Title for the plot.
        ylabel (str): Y axis label for the plot.
    """
    plt.plot(data)
    plt.title(title)
    plt.ylabel(ylabel)
    plt.savefig(savename, bbox_inches='tight')
    plt.close()
