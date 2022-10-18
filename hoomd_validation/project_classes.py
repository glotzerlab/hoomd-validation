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
        list_sps = []
        num_particles = 12**3
        replicate_indices = range(16)
        params_list = [(1.5, 0.6, 1.0270905797770546), (1.0, 0.8, 1.4363805638963822), (1.25, 0.05, 0.05363574413661169)]
        for kT, density, pressure in params_list:
            for idx in replicate_indices:
                list_sps.append({
                    "kT": kT,
                    "density": density,
                    "pressure": pressure,
                    "num_particles": num_particles,
                    "replicate_idx": idx
                })
        return list_sps

    def job_document_params(self, job):
        """list(tuple): A list of tuples (param, default) giving the job \
        document parameters and default values for this project."""
        params_list = []

        return params_list
