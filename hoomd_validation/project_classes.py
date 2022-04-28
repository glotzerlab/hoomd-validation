import signac

from abc import abstractmethod
from flow import FlowProject


class ValidationTestProject(FlowProject):
    """Base class for all validation test projects."""

    @property
    @abstractmethod
    def job_document_params(self):
        """list(dict): List of dictionaries containing statepoint parameters
        for the jobs in this project."""
        pass

    @property
    @abstractmethod
    def job_statepoints(self):
        """list(tuple): List of pairs (parameter, default) giving the job
        document parameters and their default values."""
        pass



class LJFluid(ValidationTestProject):
    """Lennard Jones Fluid Phase Behavior Validation Test.

    Right now this class only provides dummy statepoint and job document
    parameters so we can test the project skeleton.
    """

    @property
    def job_statepoints(self):
        pressures = [0.1, 0.2, 0.5, 1.0]
        job_sps = []
        for press in pressures:
            job_sps.append(
                {'kT': 1.0, 'N': 2000, 'V': 400, 'P': press}
            )
        return job_sps

    @property
    def job_document_params(self):
        params_list = []
        params_list.append(
            ('random_seeds', [123, 456, 6789])
        )
        return params_list
