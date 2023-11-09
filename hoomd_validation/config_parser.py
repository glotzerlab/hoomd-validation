# Copyright (c) 2022-2023 The Regents of the University of Michigan.
# Part of HOOMD-blue, released under the BSD 3-Clause License.

"""Class for parsing config files."""

import os
import sys
from pathlib import Path

import yaml


class ConfigFile(dict):
    """Parse config files.

    Parse workflow configuration options from ``config.yaml``. See
    ``config-sample.yaml`` for documentation of all options and their defaults.

    The parsed config file is presented as a dictionary in a ConfigFile
    instance.
    """

    DEFAULT_CONFIG_PATH = str(Path(__file__).parent / 'config.yaml')

    def __init__(self, config_file_path=DEFAULT_CONFIG_PATH):
        if not os.path.exists(config_file_path):
            config = dict()
        else:
            with open(config_file_path) as file:
                config = yaml.safe_load(file)

        self['executable'] = self._parse_executable_string(config)
        self['max_cores_sim'] = int(config.get('max_cores_sim', 16))
        self['max_cores_submission'] = int(config.get('max_cores_submission', 16))
        self['max_gpus_submission'] = int(config.get('max_gpus_submission', 1))
        self['max_walltime'] = float(config.get('max_walltime', 24))
        self['short_walltime'] = float(config.get('short_walltime', 2))
        self['replicates'] = int(config.get('replicates', 32))
        self['enable_llvm'] = bool(config.get('enable_llvm', True))
        self['enable_gpu'] = bool(config.get('enable_gpu', True))

    @staticmethod
    def _parse_executable_string(config_file):
        """Search the config file and determine the executable.

        Searches the executable section of the config file and builds the string
        needed by flow's directives. If no config file is present, we use the
        python executable used to run this code.
        """
        if 'executable' not in config_file:
            return sys.executable

        return_string = ''
        executable_options = config_file['executable']
        using_container = 'singularity_container' in executable_options
        if using_container:
            return_string += (
                'singularity exec --nv '
                + executable_options.get('singularity_options', '')
                + ' '
            )
            return_string += executable_options['singularity_container'] + ' '

        return_string += executable_options.get(
            'python_exec', 'python' if using_container else sys.executable
        )
        return return_string
