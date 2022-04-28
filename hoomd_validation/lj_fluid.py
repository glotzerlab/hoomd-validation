# Copyright (c) 2022 The Regents of the University of Michigan.
# Part of HOOMD-blue, released under the BSD 3-Clause License.

from config import test_project_dict
from project_classes import LJFluid


@LJFluid.operation
def create_initial_state(job):
    print("Created Initial LJ fluid state")
    return


@LJFluid.operation
def run_simulation(job):
    print("Running LJ Fluid simulation")
    pass


@LJFluid.operation
def analyze(job):
    print("Analyzing LJ Fluid simulation")
    pass


if __name__ == "__main__":
    LJFluid.get_project(test_project_dict["LJFluid"].root_directory()).main()
