# Tips for running on OLCF Frontier

# Recommended configuration

```
max_cores_sim: 56
max_cores_submission: 7168
max_gpus_submission: 256
max_walltime: 2
enable_llvm: false
enable_gpu: true
```

## Recommended template

```
{% extends "frontier.sh" %}

{% block header %}
    {{- super () -}}
#SBATCH -C nvme
{% endblock header %}
{% block custom_content %}

echo "Loading software environment."

export GLOTZERLAB_SOFTWARE_ROOT=/mnt/bb/${USER}/software
time srun --ntasks-per-node 1 mkdir ${GLOTZERLAB_SOFTWARE_ROOT}
time srun --ntasks-per-node 1 tar --directory ${GLOTZERLAB_SOFTWARE_ROOT} -xpf ${MEMBERWORK}/mat110/software.tar
source ${GLOTZERLAB_SOFTWARE_ROOT}/variables.sh

{% endblock custom_content %}
{% block body %}
    {{- super () -}}

echo "Completed job in $SECONDS seconds"
{% endblock body %}
```
