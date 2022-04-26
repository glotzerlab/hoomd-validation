import signac
from pathlib import Path

# variables for subprojects
test_project_list = ['hard_disks', 'lj_fluid']
test_project_dict = dict()

# path to all the project workspace directories
project_root = Path(__file__).parent.parent

# global project which manages all other projects
all_validation_tests = signac.get_project(
    str(project_root / "all_validation_tests")
)

# project for each validation test suite
for test_project in test_project_list:
    test_project_dict[test_project] = signac.get_project(
        str(project_root / test_project)
    )


