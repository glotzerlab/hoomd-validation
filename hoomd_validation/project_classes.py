import signac


class LJFluid:

    def __init__(self, signac_project_path):
        self._pr_path = signac_project_path

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
    def job_document_parameters(self):
        params_list = []
        params_list.append(
            ('random_seeds': [123, 456, 6789])
        )
        return params_list

    @property
    def project_file_name(self):
        return "lj_fluid.py"

    @property
    def signac_project(self):
        return signac.get_project(self._pr_path)
