import signac
import project_classes
from pathlib import Path


# variables for subprojects
test_project_name_list = ['HardDisks', 'LJFluid']
test_project_dict = dict()

# path to all the project workspace directories
project_root = Path(__file__).parent.parent

# global project which manages all other projects
all_validation_tests = signac.get_project(
    str(project_root / "AllValidationTests")
)

# project for each validation test suite
for project_name_str in test_project_list:
    test_project_dict[project_name_str] = \
        getattr(project_classes, project_name_str)(
            str(project_root / project_name_str)
        )


