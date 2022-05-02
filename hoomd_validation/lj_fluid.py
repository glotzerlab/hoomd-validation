# Copyright (c) 2022 The Regents of the University of Michigan.
# Part of HOOMD-blue, released under the BSD 3-Clause License.

"""Lennard Jones phase behavior validation test.

This workflow is a dummy and will be fully implemented at a later date.
"""

from config import test_project_dict
from project_classes import LJFluid


@LJFluid.operation
def create_initial_state(job):
    """Create initial system configuration."""
    print("Created initial LJ fluid state")
    return


@LJFluid.operation
def run_simulation(job):
    """Run the simulation."""
    print("Running LJ fluid simulation")
    pass


@LJFluid.operation
def analyze(job):
    """Analyze the output."""
    print("Analyzing LJ fluid simulation")
    pass


if __name__ == "__main__":
    LJFluid.get_project(test_project_dict["LJFluid"].root_directory()).main()
