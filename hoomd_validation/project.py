# Copyright (c) 2022 The Regents of the University of Michigan.
# Part of HOOMD-blue, released under the BSD 3-Clause License.

"""Project workflow entry point."""

import config
from project_class import Project
import flow

# Define subproject flow operations
import alj_2d
import lj_fluid
import hard_disk
import hard_sphere

# use srun on delta (mpiexec fails on multiple nodes)
flow.environments.xsede.DeltaEnvironment.mpi_cmd = "srun"

__all__ = [
    "alj_2d",
    "lj_fluid",
    "hard_disk",
    "hard_sphere",
]

if __name__ == "__main__":
    Project.get_project(config.project_root).main()
