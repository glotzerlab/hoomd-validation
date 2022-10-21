python3 hoomd_validation/lj_fluid.py submit -o langevin_md_cpu
python3 hoomd_validation/lj_fluid.py submit -o nvt_md_cpu
python3 hoomd_validation/lj_fluid.py submit -o npt_md_cpu
python3 hoomd_validation/lj_fluid.py submit -o nvt_mc_cpu
python3 hoomd_validation/lj_fluid.py submit -o npt_mc_cpu

python3 hoomd_validation/lj_fluid.py submit -o langevin_md_gpu --partition gpu
python3 hoomd_validation/lj_fluid.py submit -o nvt_md_gpu --partition gpu
python3 hoomd_validation/lj_fluid.py submit -o npt_md_gpu --partition gpu
python3 hoomd_validation/lj_fluid.py submit -o nvt_mc_gpu --partition gpu
