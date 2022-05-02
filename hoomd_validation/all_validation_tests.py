# Copyright (c) 2022 The Regents of the University of Michigan.
# Part of HOOMD-blue, released under the BSD 3-Clause License.

"""Signac workflow for the manager project."""

from flow import FlowProject
from config import all_validation_tests, test_project_dict

import lj_fluid

__all__ = ["lj_fluid"]


class AllValidationTests(FlowProject):
    """FlowProject class for the manager project."""
    pass


@AllValidationTests.operation
def run_all_workflows(job):
    """Run the workflow for all validation test projects."""
    pr = test_project_dict[job.sp.project_name].get_project(job.sp.path)
    pr.main()


if __name__ == "__main__":
    AllValidationTests.get_project(all_validation_tests.root_directory()).main()
