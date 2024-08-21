# Copyright (c) 2022-2024 The Regents of the University of Michigan.
# Part of HOOMD-blue, released under the BSD 3-Clause License.

"""This file contains all custom actions needed for this project."""

try:
    import hoomd

    class ComputeDensity(hoomd.custom.Action):
        """Compute the density of particles in the system.

        The density computed is a number density.

        Args:
            N: When not None, Use N instead of the number of particles when
               computing the density.
        """

        def __init__(self, N=None):
            self.N = N

        @hoomd.logging.log
        def density(self):
            """float: The density of the system."""
            if self.N is None:
                return self._state.N_particles / self._state.box.volume

            return self.N / self._state.box.volume

        def act(self, timestep):
            """Dummy act method."""
            pass
except ModuleNotFoundError as e:
    print(f'Warning: {e}')

    # This workaround is to allow `python project.py init` to succeed in CI checks
    # without requiring a working HOOMD installation.
    class ComputeDensity:
        """Placeholder class."""

        pass
