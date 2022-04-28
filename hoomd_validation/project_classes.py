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
        """list(dict): A list of statepoints for this project."""
        pass

    @property
    @abstractmethod
    def job_document_params(self):
        """list(tuple): A list of tuples (param, default) giving the job \
        document parameters and default values for this project."""
        pass


class LJFluid(ValidationTestProject):
    """FlowProject class for the Lennard Jones fluid validation test."""

    @property
    def job_statepoints(self):
        """list(dict): A list of statepoints for this project."""
        pressures = [0.1, 0.2, 0.5, 1.0]
        job_sps = []
        for press in pressures:
            job_sps.append({'kT': 1.0, 'N': 2000, 'V': 400, 'P': press})
        return job_sps

    @property
    def job_document_params(self):
        """list(tuple): A list of tuples (param, default) giving the job \
        document parameters and default values for this project."""
        params_list = []
        params_list.append(('random_seeds', [123, 456, 6789]))
        return params_list
