from config import all_validation_tests, test_project_dict

# TODO make sure directories exists for each of the projects

# make the statepoint of the global project jobs be names and paths to valiation
# test projects
for project_name_str, project_class in test_project_dict.items():
    global_project_job = all_validation_tests.open_job(
        dict(project_name=project_name_str,
             path=test_project.root_directory())
    )
    if global_project_job not in all_validation_tests:
        global_project_job.init()


# open jobs for all projects corresponding to validation test suites
for _, project_class in test_project_dict.items():

    # get the signac project for this validation test
    pr = project_class.signac_project

    # add all the jobs to the validation test project
    job_sps = project_class.job_statepoints
    for job_sp in job_sps:
        job = pr.open_job(job_sp)
        if job not in pr

        # initialize job document parameters for this validation test job
        for param, default in project_class.job_document_parameters:
            if param not in job.doc:
                job.doc.param = default
