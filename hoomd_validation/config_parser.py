import os
import json
import subprocess


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
                "containerized": true,
                "container_path": "/path/to/my/container",
                "path_to_python": "/usr/bin/python"
            }
        }
    """

    DEFAULT_CONFIG_PATH = "config.json"

    def __init__(self, config_file_path=DEFAULT_CONFIG_PATH):
        self._config_path = config_file_path

    def parse_executable_string(self):
        """Search the config file and determine the executable.

        Searches the executable section of the config file and builds the string
        needed by flow's directives.

        If no config file is present, we use the python executable on the PATH.
        """

        if not os.path.exists(self._config_path):
            print(f"Could not find config file located at {self._config_path}, "
                  "using the default python executable instead")
            python_exec = subprocess.run(["which", "python"], stdout=subprocess.PIPE, text=True)
            return python_exec.stdout[:-1] # remove the \n at the end

        return_string = ""
        with open(self._config_path) as f:
            config_file = json.load(f)
            executable_options = config_file["executable"]

            if executable_options["containerized"]:
                return_string += "singularity exec --nv "
                if len(executable_options["container_path"]) > 0:
                    return_string += (executable_options["container_path"] + " ")
                else:
                    raise RuntimeError("If containerized is true, a non-empty"
                                       " container_path must be specified")
            return_string += executable_options["path_to_python"]
        return return_string


