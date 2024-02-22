# Copyright (c) 2022-2024 The Regents of the University of Michigan.
# Part of HOOMD-blue, released under the BSD 3-Clause License.

"""Initialize signac projects and create global variables."""

from pathlib import Path

from config_parser import ConfigFile

# path to container image for simulations
CONFIG = ConfigFile()

# Path to project root directory.
project_root = Path(__file__).parent.parent
