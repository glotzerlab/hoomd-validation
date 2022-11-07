# Copyright (c) 2022 The Regents of the University of Michigan.
# Part of HOOMD-blue, released under the BSD 3-Clause License.

"""Project workflow entry point."""

import config
from project_class import Project

# Define subproject flow operations
import lj_fluid
import hard_disk

__all__ = [
    "lj_fluid",
    "hard_disk",
]

if __name__ == "__main__":
    Project.get_project(config.project_root).main()
