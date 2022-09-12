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
        num_particles = 10000
        replicate_indices = range(4)
        params_list = [(1.5, 0.6)]
        for kT, density in params_list:
            for idx in replicate_indices:
                list_sps.append({
                    "kT": kT,
                    "density": density,
                    "num_particles": num_particles,
                    "replicate_idx": idx
                })
        return list_sps

    def job_document_params(self, job):
        """list(tuple): A list of tuples (param, default) giving the job \
        document parameters and default values for this project."""
        params_list = []

        # random seeds
        id_str = str(int('0x' + job.id, base=16))
        params_list.append(('seed', int(id_str[:5])))

        # store values needed for each simulation to be run
        params_list.append(('nvt_md',
                            dict(pressure=None,
                                 aggregate_pressure=None,
                                 potential_energy=None)))
        params_list.append(('npt_md', dict(density=None,
                                           potential_energy=None)))
        params_list.append(('nvt_mc', dict(pressure=None,
                                           potential_energy=None)))
        params_list.append(('npt_mc', dict(density=None,
                                           potential_energy=None)))

        return params_list
