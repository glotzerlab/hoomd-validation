# Copyright (c) 2022 The Regents of the University of Michigan.
# Part of HOOMD-blue, released under the BSD 3-Clause License.

"""Populate signac projects with jobs and job document parameters."""

from config import all_validation_tests, test_project_dict

# initialize jobs for the manager project
for project_name_str, project_class in test_project_dict.items():
    manager_project_job = all_validation_tests.open_job(
        dict(project_name=project_name_str,
             path=project_class.path))
    if manager_project_job not in all_validation_tests:
        manager_project_job.init()

# initialize jobs for validation test projects
for _, project_class in test_project_dict.items():

    # add all the jobs to the project
    job_sps = project_class.job_statepoints
    for job_sp in job_sps:
        job = project_class.open_job(job_sp)
        if job not in project_class:
            job.init()

        # initialize job document parameters for this job
        for param, default in project_class.job_document_params(job):
            if param not in job.doc:
                setattr(job.doc, param, default)
