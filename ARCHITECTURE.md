# Architecture

## Design Goals

The code in this repoitory is designed to run validation tests using HOOMD-blue
for longer periods of time than CI testing can handle. Users of this repository
should be able to:

1. Run a set of validation test workflows on a variety of hardware setups

2. Choose specific validation workflows to run, and be able to select subsets of
the jobs in one workflow to run

3.

## Implementation

Each validation test workflow is defined in a separate signac project. There is
one global project `AllValidationTests` for which the statepoint parameters are
the name of the individual validation test as well as the path to the project
directories.

## Initialization

1. Run the shell script (still need to make this) to initialize the project
directories.

2. Run `init.py` to populate the project directories with jobs and initialize
job documents

3. Use the python files in `hoomd_validation` to run specific validation
test workflows
