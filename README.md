# HOOMD-blue Validation

This repository contains longer running validation tests for HOOMD-blue. The
validation test workflows in this repository are organized into signac projects.

## Requirements

* signac >=1.8.0
* signac-flow >= 1.22.0
* Simulation workflow steps require either the [glotzerlab-software container]
  or the following software:
    * HOOMD-blue >=3.0 *(with MPI, GPU, and LLVM support enabled)*
* Analysis workflow steps require either the [glotzerlab-software container] or
  the following software:
    * matplotlib
    * gsd
    * numpy
    * scipy
* Workstation or HPC system with at least 8 CPU cores and 1 GPU supported by
  HOOMD-blue.

## Preparation

Clone this repository:

```bash
$ git clone https://github.com/glotzerlab/hoomd-validation.git
$ cd hoomd-validation
```

## Configuration

Install the prerequisites into a Python environment of your choice. To use the
[glotzerlab-software container], create the file `hoomd_validation/config.json`
with the following contents:
```
{
    "executable": {
        "container_path": "<path-to>/software.sif"
    }
}
```
and replace `<path-to>` with the absolute path to the directory containing
`software.sif`. Add any options before the path, such as
`--bind /scratch,/gpfs <path-to>/software.sif`.

See `config_parser.py` for full documentation on all options in the config file.

## Usage

1. Initialize the signac project directories, populate them with jobs and job
documents:
    ```bash
    python3 hoomd_validation/init.py
    ```
2. Run and analyze all validation tests:
    * On a workstation (this takes a long time to complete):
        ```
        $ python all_validation_tests.py run
        ```
    * On a cluster:
        1. Populate the flow script template(s) or your shell environment appropriately.
        2. Create the simulation initial states:
            ```
            $ python hoomd_validation/lj_fluid.py submit -o create_initial_state
            ```
            *(wait for all jobs to complete)*
        3. Run the simulations (adjust partition names according to your cluster)
            ```
            $ python3 hoomd_validation/lj_fluid.py submit -o '.*_cpu' --partition standard
            $ python3 hoomd_validation/lj_fluid.py submit -o '.*_gpu' --partition gpu
            ```
            *(wait for all jobs to complete)*
        4. Run the analysis (assuming you have the analysis workflow prerequisites in your Python environment):
            ```
            $ python hoomd_validation/lj_fluid.py run
            ```
            *(alternately, submit the analysis in stages until no jobs remain eligible)*
3. Inspect the plots produced in:
    * `LJFluid/*.svg`

[glotzerlab-software container]: https://glotzerlab-software.readthedocs.io/
