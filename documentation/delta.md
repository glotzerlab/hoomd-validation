# Tips for running on NCSA Delta

# Recommended configuration

```
max_cores_sim: 64
max_cores_submission: 512
max_gpus_submission: 8
max_walltime: 48
```

# Compiling HOOMD from source

* When building with `ENABLE_LLVM=on`, build separate CPU and GPU builds in:
  * `/scratch/bbgw/${USER}/build/hoomd-cpu`
  * and `/scratch/bbgw/${USER}/build/hoomd-gpu`.
* To link to `libcuda.so`, compile `hoomd-gpu` in an interactive job:
  `srun --account=bbgw-delta-gpu --partition=gpuA40x4 --nodes=1 --tasks=1 --tasks-per-node=1 --cpus-per-task=16 --mem=48g --gpus=1 --pty zsh`

* Submitting jobs

Unset your accounts in `signac.rc` and use environment variables to choose the account and
hoomd build at submission time:

CPU:
```
SBATCH_ACCOUNT=bbgw-delta-cpu PYTHONPATH=/scratch/bbgw/${USER}/build/hoomd-cpu SBATCH_EXPORT=PYTHONPATH python hoomd_validation/project.py submit -o '.*_cpu'
```

GPU:
```
SBATCH_ACCOUNT=bbgw-delta-gpu PYTHONPATH=/scratch/bbgw/${USER}/build/hoomd-gpu SBATCH_EXPORT=PYTHONPATH python hoomd_validation/project.py submit -o '.*_gpu' --partition gpuA100x4
```
