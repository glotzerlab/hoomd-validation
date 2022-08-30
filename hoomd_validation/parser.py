import os
import json


CONFIG_FILE_PATH = "config.json"


def parse_config_file():
    """Search the config file and determine the executable.

    If no config file is present, we assume a default executable, which is
    just "python".
    """

    if not os.path.exists(CONFIG_FILE_PATH):
        return "python"

    return_string = ""
    with open(CONFIG_FILE_PATH) as f:
        config_file = json.load(f)

        if config_file["containerized"]:
            return_string += "singularity exec --nv "
            if len(config_file["container_path"]) > 0:
                return_string += config_file["container_path"]
            else:
                raise RuntimeError("If containerized is true, a non-empty"
                                   " container_path must be specified")
        return_string += config_file["path_to_python"]
    return return_string


