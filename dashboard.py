# Copyright (c) 2022-2023 The Regents of the University of Michigan.
# Part of HOOMD-blue, released under the BSD 3-Clause License.

"""Dashboard."""

from signac_dashboard import Dashboard
from signac_dashboard.modules import ImageViewer, StatepointList

modules = [
    StatepointList(),
    ImageViewer(context='JobContext', img_globs=('*.png', '*.jpg', '*.gif', '*.svg')),
    ImageViewer(
        context='ProjectContext', img_globs=('*.png', '*.jpg', '*.gif', '*.svg')
    ),
]


class ValidationDashboard(Dashboard):
    """Dashboard application."""

    def job_title(self, job):
        """Name jobs."""
        if job.statepoint.subproject == 'lj_fluid':
            return (
                f'lj_fluid: kT={job.statepoint.kT}, '
                f'rho={job.statepoint.density}, '
                f'N={job.statepoint.num_particles}'
            )

        if job.statepoint.subproject == 'lj_union':
            return f'lj_union: kT={job.statepoint.kT}, rho={job.statepoint.density}'

        if job.statepoint.subproject == 'alj_2d':
            return f'alj_2d: kT={job.statepoint.kT}, rho={job.statepoint.density}'

        if job.statepoint.subproject in (
            'hard_disk',
            'hard_sphere',
            'simple_polygon',
            'patchy_particle_pressure',
        ):
            return f'{job.statepoint.subproject}: rho={job.statepoint.density}'

        raise RuntimeError('Unexpected job')

    def job_sorter(self, job):
        """Sort jobs."""
        if job.statepoint.subproject == 'patchy_particle_pressure':
            return (
                job.statepoint.subproject,
                job.statepoint.density,
                job.statepoint.pressure,
                job.statepoint.temperature,
                job.statepoint.chi,
                job.statepoint.replicate_idx,
            )

        return (job.statepoint.subproject, job.statepoint.num_particles,)


if __name__ == '__main__':
    ValidationDashboard(modules=modules).main()
