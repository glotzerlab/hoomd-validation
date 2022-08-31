import os
import sys
import json


class ConfigFileParser:
    """Parse config files.

    This class is used to parse files named config.json, which have settings
    for configuring the execution of the validation workflows. At the moment,
    the only configuration needed is for the executable command used by flow
    to give as a directive for job operations. An example config file is shown
    below:

    .. json::

        {
            "executable": {
                "container_path": "/path/to/my/container",
                "python_exec": "/usr/bin/python"
            }
        }

    Neither field in executable is strictly required, but if working in a
    containerized environment, container_path must specify the location of the
    container. If python_exec is not specified, the default behavior is to use
    the executable being used to run the code. In a containerized environment, the
    default behavior is to use "python", since we cannot know a priori the name
    of the executable in the container.
    """

    DEFAULT_CONFIG_PATH = "config.json"

    def __init__(self, config_file_path=DEFAULT_CONFIG_PATH):
        self._config_path = config_file_path

    def parse_executable_string(self):
        """Search the config file and determine the executable.

        Searches the executable section of the config file and builds the string
        needed by flow's directives.

        If no config file is present, we use the python executable used to run
        this code.
        """

        if not os.path.exists(self._config_path):
            return sys.executable

        return_string = ""
        with open(self._config_path) as f:
            config_file = json.load(f)

            if "executable" not in config_file:
                return sys.executable

            executable_options = config_file["executable"]
            using_container = "container_path" in executable_options
            if using_container:
                return_string += "singularity exec --nv "
                return_string += (executable_options["container_path"] + " ")

            if "python_exec" in executable_options:
                return_string += executable_options["python_exec"]
            else:
                return_string += ("python" if using_container else sys.executable)
        return return_string


