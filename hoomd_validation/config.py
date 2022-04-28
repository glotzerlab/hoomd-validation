# Copyright (c) 2022 The Regents of the University of Michigan.
# Part of HOOMD-blue, released under the BSD 3-Clause License.

import signac
import project_classes
from pathlib import Path

# variables for subprojects
test_project_name_list = ['LJFluid']
test_project_dict = dict()

# path to all the project workspace directories
project_root = Path(__file__).parent.parent

# global project which manages all other projects
all_validation_tests = signac.init_project(name="AllValidationTests",
                                           root=str(project_root
                                                    / "AllValidationTests"))

# project for each validation test suite
for project_name_str in test_project_name_list:
    test_project_dict[project_name_str] = \
        getattr(project_classes, project_name_str).init_project(
            name=project_name_str,
            root=str(project_root / project_name_str)
        )
