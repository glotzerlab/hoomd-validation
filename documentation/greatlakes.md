# Tips for running on UMich Great Lakes

# Recommended configuration

```
max_cores_sim: 32
max_cores_submission: 32
max_gpus_submission: 1
max_walltime: 96
```

# Compiling HOOMD from source

* When building with `ENABLE_LLVM=on`, built separate CPU and GPU builds in:
  * `${HOME}/build/hoomd-cpu`
  * and ``${HOME}/build/hoomd-gpu`.
* To link to `libcuda.so`, compile `hoomd-gpu` in an interactive job:
  `srun -Asglotzer<N> --gres=gpu:1 --nodes=1 --ntasks-per-node=1 --cpus-per-task=8 --partition gpu -t 8:00:00 --mem=64G --pty /bin/zsh`

* Submitting jobs

Set environment variables to choose the hoomd build and memory requirement at submission time:

CPU:
```
SBATCH_MEM_PER_CPU="4g" PYTHONPATH=${HOME}/build/hoomd-cpu SBATCH_EXPORT=PYTHONPATH python3 hoomd_validation/project.py submit -o '.*_cpu'
```

GPU:
```
SBATCH_MEM_PER_CPU="64g" PYTHONPATH=${HOME}/build/hoomd-gpu SBATCH_EXPORT=PYTHONPATH python hoomd_validation/project.py submit -o '.*_gpu' --partition gpu
```
