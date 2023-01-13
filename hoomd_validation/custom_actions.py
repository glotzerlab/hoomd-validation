# Copyright (c) 2022-2023 The Regents of the University of Michigan.
# Part of HOOMD-blue, released under the BSD 3-Clause License.

"""This file contains all custom actions needed for this project."""

import hoomd


class ComputeDensity(hoomd.custom.Action):
    """Compute the density of particles in the system.

    The density computed is a number density.
    """

    @hoomd.logging.log
    def density(self):
        """float: The density of the system."""
        return self._state.N_particles / self._state.box.volume

    def act(self, timestep):
        """Dummy act method."""
        pass
