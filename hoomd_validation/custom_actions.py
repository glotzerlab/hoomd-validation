# Copyright (c) 2022 The Regents of the University of Michigan.
# Part of HOOMD-blue, released under the BSD 3-Clause License.

"""This file contains all custom actions needed for this project."""

import hoomd


class ComputeDensity(hoomd.custom.Action):
    """Compute the density of particles in the system.

    The density computed is a number density.
    """

    def __init__(self, sim, num_particles):
        self._sim = sim
        self._num_particles = num_particles

    @hoomd.logging.log
    def density(self):
        """float: The density of the system."""
        vol = None
        with self._sim.state.cpu_local_snapshot as snap:
            vol = snap.global_box.volume
        return self._num_particles / vol

    def act(self, timestep):
        """Dummy act method."""
        pass
