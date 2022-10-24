# Copyright (c) 2022 The Regents of the University of Michigan.
# Part of HOOMD-blue, released under the BSD 3-Clause License.

"""Signac workflow for the manager project."""

import config
from project_class import Project

# Define subproject flow operations
import lj_fluid

__all__ = ["lj_fluid"]

if __name__ == "__main__":
    Project.get_project(config.project_root).main()
