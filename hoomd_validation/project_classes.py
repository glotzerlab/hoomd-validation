# Copyright (c) 2022 The Regents of the University of Michigan.
# Part of HOOMD-blue, released under the BSD 3-Clause License.

"""FlowProject classes for each validation test."""

from abc import abstractmethod
from flow import FlowProject


class ValidationTestProject(FlowProject):
    """Base class for all validation test FlowProjects."""

    @property
    @abstractmethod
    def job_statepoints(self):
        """list(dict): A list of statepoints for this project.

        Used to instantiate jobs for subclassed projects.
        """
        pass

    @abstractmethod
    def job_document_params(self, job):
        """list(tuple): A list of tuples (param, default) giving the job \
        document parameters and default values for this project."""
        pass


class LJFluid(ValidationTestProject):
    """FlowProject class for the Lennard Jones fluid validation test.

    Right now this class only provides dummy statepoint and job document
    parameters so we can test the project skeleton.
    """

    @property
    def job_statepoints(self):
        """list(dict): A list of statepoints for this project."""
        num_particles = 12**3
        replicate_indices = range(4)
        params_list = [(1.5, 0.5998286671851658, 1.0270905797770546),
                       (1.0, 0.7999550814681395, 1.4363805638963822),
                       (1.25, 0.049963649769543844, 0.05363574413661169)]
        for kT, density, pressure in params_list:
            for idx in replicate_indices:
                yield ({
                    "kT": kT,
                    "density": density,
                    "pressure": pressure,
                    "num_particles": num_particles,
                    "replicate_idx": idx
                })

    def job_document_params(self, job):
        """list(tuple): A list of tuples (param, default) giving the job \
        document parameters and default values for this project."""
        return []
