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


def get_energies(traj):
    """Get potential energies from a gsd trajectory.

    Args:
        traj (`gsd.hoomd.Trajectory`): trajectory to read data from
    """
    return get_log_quantity(
        traj, 'md/compute/ThermodynamicQuantities/potential_energy')


def get_pressures(traj):
    """Get pressures from a gsd trajectory.

    Args:
        traj (`gsd.hoomd.Trajectory`): trajectory to read data from
    """
    return get_log_quantity(traj, 'md/compute/ThermodynamicQuantities/pressure')


def plot_quantity(data, savename, title, ylabel):
    """Plot a quanity using matplotlib.

    Args:
        data (`np.ndarray` (N, )): One dimensional data to plot.
        savename (str): File path for saving the plot.
        title (str): Title for the plot.
        ylabel (str): Y axis label for the plot.
    """
    plt.plot(data)
    plt.title(title)
    plt.ylabel(ylabel)
    plt.savefig(savename, bbox_inches='tight')
    plt.close()


def plot_pressures(data, save_filename):
    """Plot pressures using matplotlib.

    Args:
        data (`np.ndarray` (N, )): One dimensional data to plot.
        savename (str): File path for saving the plot.
    """
    plot_quantity(data, save_filename, title='Pressure vs. time', ylabel='$P$')


def plot_energies(data, save_filename):
    """Plot potential energies using matplotlib.

    Args:
        data (`np.ndarray` (N, )): One dimensional data to plot.
        savename (str): File path for saving the plot.
    """
    plot_quantity(data,
                  save_filename,
                  title='Potential Energy vs. time',
                  ylabel='$U$')


def plot_densities(data, save_filename):
    """Plot densities using matplotlib.

    Args:
        data (`np.ndarray` (N, )): One dimensional data to plot.
        savename (str): File path for saving the plot.
    """
    plot_quantity(data,
                  save_filename,
                  title='Number Density vs. time',
                  ylabel='$\\rho$')
