# Copyright (c) 2022-2023 The Regents of the University of Michigan.
# Part of HOOMD-blue, released under the BSD 3-Clause License.

"""Populate the signac project with jobs and job document parameters."""

# import subprojects
import alj_2d
import config
import hard_disk
import hard_sphere
import lj_fluid
import lj_union
import patchy_particle_pressure
import signac
import simple_polygon

subprojects = [
    # alj_2d,
    lj_fluid,
    # lj_union,
    # hard_disk,
    # hard_sphere,
    # simple_polygon,
    # patchy_particle_pressure,
]

project = signac.init_project(path=config.project_root)

# initialize jobs for validation test projects
for subproject in subprojects:
    # add all the jobs to the project
    for job_sp in subproject.job_statepoints():
        job = project.open_job(job_sp).init()
