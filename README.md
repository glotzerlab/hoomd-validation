# HOOMD-blue Validation

This repository contains validation tests for HOOMD-blue. The workflows are organized in
a [signac] workspace and use [row].

## Preparation

Clone this repository:
```bash
git clone https://github.com/glotzerlab/hoomd-validation.git
```

Then change to the repository's directory:
```bash
cd hoomd-validation
```

## Configuration

1. Install the requirements (see below) into a Python environment of your choice.
2. Copy `hoomd_validation/config-sample.toml` to `hoomd_validation/config.toml`
   and set the parameters as desired. Each option is documented by a comment in the 
   sample configuration file.
3. Initialize the signac project directories and create `workflow.toml`.
    ```bash
    python3 hoomd_validation/project.py init
    ```
4. Configure [row] as necessary for your workstation or HPC resources.
   > Note: `project.py init` will overwrite `workflow.toml`. 

[row]: https://row.readthedocs.io

## Execute tests

Run
```bash
row submit
```

To submit the first stage of the workflow. Wait for all the jobs to complete, then run
`row submit` again to start the second stage. Most subprojects in the validation
workflow have 4 stages ending with `compare_modes`.

> Note: You can execute a single subproject with `row submit --action 'subproject_name.*'

After you execute `compare_mode`, inspect the `svg` files saved in the repository root.
You can also run the provided [signac-dashboard] application to explore the results in a
web browser:

```bash
python3 dashboard.py run
```

[signac]: https://signac.readthedocs.io
[signac-dashboard]: https://signac-dashboard.readthedocs.io

## Requirements

* h5py
* hoomd >= 4.6.0
* matplotlib
* numpy
* rtoml
* scipy
* signac >= 2.2.0
* signac-dashboard [optional]
