# Tips for running on OLCF Summit

# Recommended configuration

```
max_cores_sim: 42
max_cores_submission: 1344
max_gpus_submission: 192
max_walltime: 2
```

## Recommended template

* Write stdout/stderr to files.
* Unload `darshan-runtime` to prevent jobs from hanging on exit.

```
{% extends "summit.sh" %}

{% block header %}
    {{- super () -}}
#BSUB -o hoomd-validation.%J.out
#BSUB -e hoomd-validation.%J.out

{% endblock header %}
{% block custom_content %}
echo "Loading modules."
source /ccs/proj/mat110/glotzerlab-software/joaander-test/environment.sh
module unload darshan-runtime
set -x
{% endblock custom_content %}
{% block body %}
    {{- super () -}}
{% endblock body %}
```
