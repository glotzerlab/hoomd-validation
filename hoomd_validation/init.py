from config import all_validation_tests, test_project_dict

# TODO make sure directories exists for each of the projects

# make the statepoint of the global project jobs be paths to subprojects
for test_project in test_project_dict:
    global_project_job = all_validation_tests.open_job(
        dict(test_project_dict[test_project])
    )
    if global_project_job not in all_validation_tests:
        global_project_job.init()


# each test_project needs to define statepoint parameters and open all its jobs

