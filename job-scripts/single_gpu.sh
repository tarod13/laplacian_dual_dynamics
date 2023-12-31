#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=8G
#SBATCH --time=23:0:0
#SBATCH --mail-user=gomeznor@ualberta.ca
#SBATCH --mail-type=ALL

cd ~/projects/def-bowling/diegog/laplacian_dual_dynamics
module load apptainer

apptainer exec --nv -B /home -B $PWD:/pwd -B /project -B /scratch -B /localscratch -B /project/def-mbowling/diegog/laplacian_dual_dynamics/ --pwd /pwd ~/apptainer/lk_haiku.sif python3 train_laprepr.py "E2" --save_dir ~/logs/laplacian_dual_dynamics/ --config_file generalized_gdo.yaml --env_name GridMaze-19 --regularization_weight 1 --seed 1234 --use_wandb
