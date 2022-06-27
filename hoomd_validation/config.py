# Copyright (c) 2022 The Regents of the University of Michigan.
# Part of HOOMD-blue, released under the BSD 3-Clause License.

"""Initialize signac projects and create global variables."""

import signac
import project_classes
from pathlib import Path

# path to container image for simulations
CONTAINER_IMAGE_PATH = "/ocean/projects/dmr170059p/tomwalt/software.sif"

# global variables for validation test projects
test_project_name_list = ['LJFluid']
test_project_dict = dict()

project_root = Path(__file__).parent.parent

# initialize manager project
all_validation_tests = signac.init_project(name="AllValidationTests",
                                           root=str(project_root
                                                    / "AllValidationTests"))

# initialize validation test projects
for project_name_str in test_project_name_list:
    test_project_dict[project_name_str] = \
        getattr(project_classes, project_name_str).init_project(
            name=project_name_str,
            root=str(project_root / project_name_str)
        )
