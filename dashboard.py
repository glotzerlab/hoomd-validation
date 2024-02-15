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
        if job.cached_statepoint['subproject'] == 'lj_fluid':
            return (
                f'lj_fluid: kT={job.cached_statepoint["kT"]}, '
                f'rho={job.cached_statepoint["density"]}, '
                f'N={job.cached_statepoint["num_particles"]}'
            )

        if job.cached_statepoint['subproject'] == 'lj_union':
            return f'lj_union: kT={job.cached_statepoint["kT"]}, rho={job.cached_statepoint["density"]}'

        if job.cached_statepoint['subproject'] == 'alj_2d':
            return f'alj_2d: kT={job.cached_statepoint["kT"]}, rho={job.cached_statepoint["density"]}'

        if job.cached_statepoint['subproject'] in (
            'hard_disk',
            'hard_sphere',
            'simple_polygon',
            'patchy_particle_pressure',
        ):
            return f'{job.cached_statepoint["subproject"]}: rho={job.cached_statepoint["density"]}'

        raise RuntimeError('Unexpected job')

    def job_sorter(self, job):
        """Sort jobs."""
        if job.cached_statepoint['subproject'] == 'patchy_particle_pressure':
            return (
                job.cached_statepoint['subproject'],
                job.cached_statepoint['density'],
                job.cached_statepoint['pressure'],
                job.cached_statepoint['temperature'],
                job.cached_statepoint['chi'],
                job.cached_statepoint['replicate_idx'],
            )

        return (
            job.cached_statepoint['subproject'],
            job.cached_statepoint['num_particles'],
        )


if __name__ == '__main__':
    ValidationDashboard(modules=modules).main()
