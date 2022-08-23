# Copyright (c) 2022 The Regents of the University of Michigan.
# Part of HOOMD-blue, released under the BSD 3-Clause License.

"""Classes for computing pressure in LJ Fluid MC Simulations."""

import freud
import numpy as np


class LJForce:
    """Compute Lennard-Jones Forces. With Xplor smoothing.

    Args:
        sigma (float): The interaction width.
        epsilon (float): The interaction strength.
        r_cut (float): The interaction cutoff distance.
    """

    def __init__(self, sigma, epsilon, r_cut, r_on):
        self._sigma = sigma
        self._epsilon = epsilon
        self._r_on = r_on
        self._r_cut = r_cut

    @property
    def r_cut(self):
        """The cutoff radius for this interaction."""
        return self._r_cut

    def __call__(self, rijs):
        """Compute the force for each displacement vector.

        Args:
            rijs (`np.ndarray` (N, 3)): A set of displacement vectors.

        Returns:
            `np.ndarray` (N, 3): The force for each displacement vector.
        """
        distances = np.linalg.norm(rijs, axis=-1)
        uvecs = rijs / distances[:, None]

        # different regions of the potential
        outside_r_cut = distances > self._r_cut
        smoothing_region = np.logical_and(distances > self._r_on,
                                          distances < self._r_cut)
        plain_region = distances < self._r_on

        # array to populated
        forces = np.zeros_like(rijs)

        # lj potential
        U = 4 * self._epsilon * ((self._sigma / distances)**12 -
                                 (self._sigma / distances)**6)

        # dU/dr
        dUdr = -24 * self._epsilon * (2 * (self._sigma**12 / distances**13) -
                                      (self._sigma**6 / distances**7))

        # smoothing function
        rcutsq = self._r_cut**2
        ronsq = self._r_on**2
        rsq = distances**2
        S = (rcutsq - rsq) * (rcutsq + 2 * rsq - 3 * ronsq) / (rcutsq - ronsq)

        # derivative of smoothing function
        r = distances
        dSdr = 4 * r * (rcutsq - rsq) - 2 * r * (rcutsq + 2 * rsq
                                                 - 3 * ronsq) / (rcutsq - ronsq)

        # set forces
        forces[plain_region, :] = -1 * uvecs[plain_region] * dUdr[plain_region,
                                                                  None]
        forces[smoothing_region, :] = -1 * uvecs[smoothing_region] * (
            S[smoothing_region] * dUdr[smoothing_region]
            + U[smoothing_region] * dSdr[smoothing_region])[:, None]
        forces[outside_r_cut, :] = 0.0
        return forces


class PressureCompute:
    """Compute the pressure for continuous-potential simulations.

    This class computes the pressure by evaluating the virial term for the given
    pair force between particles.

    Args:
        force_eval (`LJForce`): Pair force for computing the virial.
    """

    def __init__(self, force_eval):
        self._force_eval = force_eval

    def compute(self, positions, box, kT):
        """Compute the pressure.

        Args:
            positions (`np.ndarray` (N, 3)): Position of all particles in the
                simulation.
            box (`box_like`): Simulation box. Takes any valid argument to
                `freud.box.Box.from_box`.
            kT (float): Thermodynamic temperature of the
                simulation.
        """
        # do freud neighbor search
        b = freud.box.Box.from_box(box)
        aabb = freud.locality.AABBQuery.from_system((b, positions))
        nlist = aabb.query(positions,
                           dict(r_max=self._force_eval.r_cut,
                                exclude_ii=True)).toNeighborList()

        # compute net force on each particle
        rijs_points = b.wrap(positions[nlist.point_indices]
                             - positions[nlist.query_point_indices])
        forces_points = self._force_eval(rijs_points)
        forces_query_points = -1 * forces_points
        forces = np.zeros_like(positions)
        np.add.at(forces, nlist.point_indices, forces_points)
        np.add.at(forces, nlist.query_point_indices, forces_query_points)

        # compute pressure
        virial = np.average(np.einsum("ij,ij->i", forces, positions))
        num_particles = len(positions)
        density = num_particles / b.volume
        pressure = kT * density * (1 - virial /
                                   (num_particles * b.dimensions * kT))
        return pressure
