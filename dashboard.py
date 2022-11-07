# Copyright (c) 2022 The Regents of the University of Michigan.
# Part of HOOMD-blue, released under the BSD 3-Clause License.

"""Dashboard."""

from signac_dashboard import Dashboard
from signac_dashboard.modules import StatepointList, ImageViewer

modules = [
    StatepointList(),
    ImageViewer(context="JobContext",
                img_globs=("*.png", "*.jpg", "*.gif", "*.svg")),
    ImageViewer(context="ProjectContext",
                img_globs=("*.png", "*.jpg", "*.gif", "*.svg")),
]


class ValidationDashboard(Dashboard):
    """Dashboard application."""

    def job_title(self, job):
        """Name jobs."""
        if job.statepoint.subproject == 'lj_fluid':
            return f"lj_fluid: kT={job.statepoint.kT}, " \
                   f"rho={job.statepoint.density}"
        elif job.statepoint.subproject == 'hard_disk':
            return f"{job.statepoint.subproject}: rho={job.statepoint.density}"
        else:
            raise RuntimeError("Unexpected job")


if __name__ == "__main__":
    ValidationDashboard(modules=modules).main()
