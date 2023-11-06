# Copyright (c) 2022-2023 The Regents of the University of Michigan.
# Part of HOOMD-blue, released under the BSD 3-Clause License.

"""Project workflow entry point."""

# Define subproject flow operations
import alj_2d
import config
import flow
import hard_disk
import hard_sphere
import lj_fluid
import lj_union
import patchy_particle_pressure
import simple_polygon
from project_class import Project

# use srun on delta (mpiexec fails on multiple nodes)
flow.environments.xsede.DeltaEnvironment.mpi_cmd = "srun"

__all__ = [
    "alj_2d",
    "lj_fluid",
    "lj_union",
    "hard_disk",
    "hard_sphere",
    "simple_polygon",
    "patchy_particle_pressure",
]

if __name__ == "__main__":
    Project.get_project(config.project_root).main()
