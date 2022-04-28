# HOOMD-blue Validation

This repository contains longer running validation tests for HOOMD-blue. Use
this repository to test your installation of hoomd. The validation test
workflows in this repository are organized into signac projects.

## Requirements

* HOOMD-blue v3
* signac
* signac-flow
* signac-dashboard (optional)

## Usage

1. Clone this repository:
    * `git clone https://github.com/glotzerlab/hoomd-validation.git`
    * `cd hoomd-validation/hoomd_validation`
2. Initialize the signac project directories, populate them with jobs and job
documents:
    * `python init.py`
3. Run the validation tests:
    * `python all_validation_tests.py run`

## Validation Tests

There is only one dummy validation test right now, `LJFluid`, which will be
made a full validation test in the future.
