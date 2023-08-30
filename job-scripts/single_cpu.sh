#!/bin/bash
#SBATCH --mem=16G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --time=23:0:0
#SBATCH --mail-user=gomeznor@ualberta.ca
#SBATCH --mail-type=ALL

cd ~/projects/def-bowling/diegog/laplacian_dual_dynamics
module load apptainer

apptainer exec --nv -B /home -B $PWD:/pwd -B /project -B /scratch -B /localscratch -B ~/project/def-mbowling/diegog/laplacian_dual_dynamics/ --pwd /pwd ~/apptainer/lk_haiku.sif python3 train_laprepr.py --save_dir ~/logs/laplacian_dual_dynamics/ --total_train_steps 8000000 --env_name GridRoom-64 --regularization_weight 0.04 --seed 1234
