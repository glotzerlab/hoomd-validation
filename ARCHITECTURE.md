# Architecture

## Design Goals

The code in this repository is designed to run validation tests using HOOMD-blue
for longer periods of time than CI testing can handle. Users of this repository
should be able to:

1. Run a set of validation test workflows on a variety of hardware setups.

2. Choose specific validation workflows to run, and be able to select subsets of
the operations in one workflow to run.

3. Visualize the validation test output and analysis using signac-dashboard.

## Implementation

To minimize the amount of effort needed to execute all test workflows (1),
Each validation test workflow is defined as a "subproject" of a single signac-flow
project. All operations on a subproject are prefixed with the subprojet's name
to allow for regex selection of operations at the command line (2). All operations
in a subproject use a precondition or `select` argument to limit their operations
only to the signac jobs specific to that subproject.

To further facilitate (2), all subprojects that require it will have an operation
`<subproject>_create_initial_state` as the first step in the workflow to prepare the
initial conditions used for later steps. All subprojects will also suffix operation
names with `_cpu` or `_gpu` according to the HOOMD device they execute on.

Each subproject is defined in its own module file (e.g. `lj_fluid.py`). Each module
must have a function `job_statepoints` that generates the statepoints needed for the job.
Each statepoint must have a key `"subproject"` with its name matching the subproject.
The subproject module file also includes all the flow operations for that subproject.

To add a subproject, implement its module, then:
1. Import the subproject module in `project.py`.
2. Import the subproject module in `init.py` and add it to the list of subprojects.
