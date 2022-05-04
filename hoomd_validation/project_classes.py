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

    @property
    @abstractmethod
    def job_document_params(self):
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
        params_list = [(0.75, 0.05), (0.75, 0.8), (1.0, 0.2), (1.0, 0.4),
                       (1.5, 0.6), (1.5, 0.7)]
        for kT, density in params_list:
            list_sps.append({
                "kT": kT,
                "density": density,
                "num_particles": num_particles
            })
        return list_sps

    def job_document_params(self, job):
        """list(tuple): A list of tuples (param, default) giving the job \
        document parameters and default values for this project."""
        params_list = []

        # random seeds
        id_str = str(int('0x' + job.id, base=16))
        seed0 = int(id_str[-4:])
        seed1 = int(id_str[-8:-4])
        seed2 = int(id_str[-12:-8])
        seed3 = int(id_str[-16:-12])

        # store values needed for each simulation to be run
        params_list.append(
            ('nvt_md', dict(seed=seed0, pressure=0.0, potential_energy=0.0)))
        params_list.append(
            ('npt_md', dict(seed=seed1, density=0.0, potential_energy=0.0)))
        params_list.append(
            ('nvt_mc', dict(seed=seed2, pressure=0.0, potential_energy=0.0)))
        params_list.append(
            ('npt_mc', dict(seed=seed3, density=0.0, potential_energy=0.0)))

        return params_list
