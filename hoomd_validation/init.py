# Copyright (c) 2022 The Regents of the University of Michigan.
# Part of HOOMD-blue, released under the BSD 3-Clause License.

"""Populate the signac project with jobs and job document parameters."""

import signac

import config

# import subprojects
import lj_fluid
import hard_disk
import hard_sphere

subprojects = [lj_fluid, hard_disk, hard_sphere]

project = signac.init_project(name="hoomd-validation", root=config.project_root)

# initialize jobs for validation test projects
for subproject in subprojects:

    # add all the jobs to the project
    for job_sp in subproject.job_statepoints():
        job = project.open_job(job_sp).init()
