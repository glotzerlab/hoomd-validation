# Copyright (c) 2022-2023 The Regents of the University of Michigan.
# Part of HOOMD-blue, released under the BSD 3-Clause License.

"""Populate the signac project with jobs and job document parameters."""

import signac

import config

# import subprojects
import alj_2d
import lj_fluid
import lj_union
import hard_disk
import hard_sphere

subprojects = [alj_2d, lj_fluid, lj_union, hard_disk, hard_sphere]

project = signac.init_project(root=config.project_root)

# initialize jobs for validation test projects
for subproject in subprojects:

    # add all the jobs to the project
    for job_sp in subproject.job_statepoints():
        job = project.open_job(job_sp).init()
