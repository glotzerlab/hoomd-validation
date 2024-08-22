# Copyright (c) 2022-2024 The Regents of the University of Michigan.
# Part of HOOMD-blue, released under the BSD 3-Clause License.

"""Project workflow entry point."""

# Define subproject flow operations
import alj_2d
import config
import hard_disk
import hard_sphere
import lj_fluid
import lj_union
import patchy_particle_pressure
import signac
import simple_polygon
from workflow_class import ValidationWorkflow

all_subprojects = [
    alj_2d,
    lj_fluid,
    lj_union,
    hard_disk,
    hard_sphere,
    simple_polygon,
    patchy_particle_pressure,
]


def init(args):
    """Initialize the workspace."""
    # TODO: uncomment
    # if (config.project_root / 'workspace').exists():
    #     message = "The project already initialized."
    #     raise RuntimeError(message)

    project = signac.init_project(path=config.project_root)

    # initialize jobs for validation test projects
    for subproject in all_subprojects:
        # add all the jobs to the project
        for job_sp in subproject.job_statepoints():
            project.open_job(job_sp).init()


if __name__ == '__main__':
    ValidationWorkflow.main(
        entrypoint=config.project_root / 'hoomd_validation' / 'project.py',
        init=init,
        path=config.project_root,
    )
