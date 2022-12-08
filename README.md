# HOOMD-blue Validation

This repository contains longer running validation tests for HOOMD-blue. The
validation test workflows in this repository are organized into signac projects.

## Requirements

* numpy
* signac >=1.8.0
* signac-flow >= 0.22.0
* signac-dashboard [optional]
* Simulation workflow steps require either the [glotzerlab-software container]
  or the following software:
    * HOOMD-blue >=3.0 *(with MPI and GPU support enabled, LLVM support is optional)*,
* Analysis workflow steps require either the [glotzerlab-software container] or
  the following software:
    * matplotlib
    * gsd
    * numpy
    * scipy
* Workstation or HPC system with at least 16 CPU cores and 1 GPU supported by
  HOOMD-blue.

## Preparation

Clone this repository:

```bash
$ git clone https://github.com/glotzerlab/hoomd-validation.git
$ cd hoomd-validation
```

## Configuration

Install the prerequisites into a Python environment of your choice. To use the
[glotzerlab-software container], copy `hoomd_validation/config-sample.yaml` to
`hoomd_validation/config.yaml`, uncomment the executable mapping, and set
`singularity_container` to your container image's path.

`hoomd_validation/config.yaml` also controls a number of job submission
parameters. See the commented options in `hoomd_validation/config-sample.yaml`
for a list and their default values.

## Usage

1. Initialize the signac project directories, populate them with jobs and job
documents:
    ```bash
    python3 hoomd_validation/init.py
    ```
2. Run and analyze all validation tests:
    * On a workstation (this takes a long time to complete):
        ```
        $ python hoomd_validation/project.py run
        ```
    * On a cluster:
        1. Populate the flow script template or your shell environment appropriately.
            ```
            $ flow template create
            $ vim templates/script.sh  # make changes to e.g. load modules
            ```
        2. Create the simulation initial states:
            ```
            $ python hoomd_validation/project.py submit -o '.*create_initial_state'
            ```
            *(wait for all jobs to complete)*
        3. Run the simulations (adjust partition names according to your cluster)
            ```
            $ python3 hoomd_validation/project.py submit -o '.*_cpu' --partition standard
            $ python3 hoomd_validation/project.py submit -o '.*_gpu' --partition gpu
            ```
            *(wait for all jobs to complete)*
        4. Run the analysis (assuming you have the analysis workflow prerequisites in your Python environment):
            ```
            $ python hoomd_validation/project.py run
            ```
            *(alternately, submit the analysis in stages until no jobs remain eligible)*
3. Inspect the plots produced in:
    * `workspace/*.svg`

## Dashboard

Run the provided [signac-dashboard] application to explore the results in a web browser:

```bash
$ python3 dashboard.py run
```

[glotzerlab-software container]: https://glotzerlab-software.readthedocs.io/
[signac-dashboard]: https://docs.signac.io/projects/dashboard/
