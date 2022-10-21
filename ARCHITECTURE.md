# Architecture

## Design Goals

The code in this repository is designed to run validation tests using HOOMD-blue
for longer periods of time than CI testing can handle. Users of this repository
should be able to:

1. Run a set of validation test workflows on a variety of hardware setups

2. Choose specific validation workflows to run, and be able to select subsets of
the operations in one workflow to run

3. Visualize the validation test output and analysis using signac-dashboard

## Implementation

Each validation test workflow is defined in a separate signac project. There is
one manager project `AllValidationTests` for which the statepoint parameters are
the name of the individual validation tests as well as the path to their project
directories.

Each validation test project must have a class defined in `project_classes.py`
which defines the job statepoints and job document parameters for that
validation test. Each validation test must also write the test workflow in a
corresponding file in the `hoomd_validation` directory.
