# Copyright (c) 2022 The Regents of the University of Michigan.
# Part of HOOMD-blue, released under the BSD 3-Clause License.

import signac

from abc import abstractmethod
from flow import FlowProject


class ValidationTestProject(FlowProject):

    @property
    @abstractmethod
    def job_document_params(self):
        pass

    @property
    @abstractmethod
    def job_statepoints(self):
        pass


class LJFluid(ValidationTestProject):

    @property
    def job_statepoints(self):
        pressures = [0.1, 0.2, 0.5, 1.0]
        job_sps = []
        for press in pressures:
            job_sps.append({'kT': 1.0, 'N': 2000, 'V': 400, 'P': press})
        return job_sps

    @property
    def job_document_params(self):
        params_list = []
        params_list.append(('random_seeds', [123, 456, 6789]))
        return params_list
