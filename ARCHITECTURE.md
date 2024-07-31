# Architecture

## Design Goals

The code in this repository is designed to run validation tests using HOOMD-blue
for longer periods of time than CI testing can handle. Users of this repository
should be able to:

1. Run a set of validation test workflows on the machine of their choice (workstation
   and/or HPC).

2. Choose specific validation workflows to run and be able to select subsets of
   the actions in one workflow to run.

3. Visualize the test output and analysis using signac-dashboard.

## Implementation

To minimize the amount of effort needed to execute all test workflows (1),
Each validation test workflow is defined as a "subproject" of a single row
project. All actions on a subproject are prefixed with the subprojet's name
to allow for glob selection of actions at the command line (2). All actions
in a subproject limit their actions to the signac jobs specific to that
subproject.

To further facilitate (2), all subprojects that require it will have an action
`<subproject>.create_initial_state` as the first step in the workflow to prepare the
initial conditions used for later steps. All subprojects will also suffix action
names with `_cpu` or `_gpu` according to the HOOMD device they execute on.

Each subproject is defined in its own module file (e.g. `lj_fluid.py`). Each module
must have a function `job_statepoints` that generates the state points needed for the
job. Every state point must have a key `"subproject"` with its name matching the
subproject. The subproject module file implements all the actions.

To add a subproject, implement its module, then:
1. Import the subproject module in `project.py`.
2. Import the subproject module in `init.py` and add it to the list of subprojects.

## Configuration

`hoomd-validation` allows user configuration of many parameters (such as walltime,
cores per job, etc...). Therefore, the row `workflow.toml` file must be dynamically
generated which is facilitated by the module `workflow.py`. Each subproject file (e.g.
`lj_fluid.py`) adds actions to the global list of actions in `action.py` with the
computed parameters based on the configuration file. The list of actions is used in
two ways. First, `init.py` will write out the `workflow.toml` that corresponds to
the current configuration. Second, `project.py` will dispatch actions to the methods
registered in `action.py`.
