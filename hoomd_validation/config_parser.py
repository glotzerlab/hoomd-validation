# Copyright (c) 2022-2024 The Regents of the University of Michigan.
# Part of HOOMD-blue, released under the BSD 3-Clause License.

"""Class for parsing config files."""

import os
from pathlib import Path

import rtoml


class ConfigFile(dict):
    """Parse config files.

    Parse workflow configuration options from ``config.yaml``. See
    ``config-sample.yaml`` for documentation of all options and their defaults.

    The parsed config file is presented as a dictionary in a ConfigFile
    instance.
    """

    DEFAULT_CONFIG_PATH = str(Path(__file__).parent / 'config.toml')

    def __init__(self, config_file_path=DEFAULT_CONFIG_PATH):
        if not os.path.exists(config_file_path):
            config = dict()
        else:
            with open(config_file_path, encoding='utf-8') as file:
                config = rtoml.load(file)

        self['max_cores_sim'] = int(config.get('max_cores_sim', 16))
        self['max_cores_submission'] = int(config.get('max_cores_submission', 16))
        self['max_gpus_submission'] = int(config.get('max_gpus_submission', 1))
        self['max_walltime'] = str(config.get('max_walltime', '1 day, 00:00:00'))
        self['short_walltime'] = str(config.get('short_walltime', '02:00:00'))
        self['replicates'] = int(config.get('replicates', 32))
        self['enable_gpu'] = bool(config.get('enable_gpu', True))
