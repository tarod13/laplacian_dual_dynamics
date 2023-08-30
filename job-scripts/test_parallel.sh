#!/bin/bash
#SBATCH --mem=16G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=16
#SBATCH --time=23:0:0
#SBATCH --mail-user=gomeznor@ualberta.ca
#SBATCH --mail-type=ALL

cd ~/projects/def-bowling/diegog/laplacian_dual_dynamics
module load apptainer

parallel apptainer exec --nv -B /home -B $PWD:/pwd -B /project -B /scratch -B /localscratch -B /project/def-mbowling/diegog/laplacian_dual_dynamics/ --pwd /pwd ~/apptainer/lk_haiku.sif python3 train_laprepr.py --save_dir ~/logs/laplacian_dual_dynamics/ --env_name GridRoom-64 --seed {1} --regularization_weight {2} ::: 1234 666 6572 8642 ::: 0.2 1.0 5.0 25.0
