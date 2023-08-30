#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24    # There are 24 CPU cores on P100 Cedar GPU nodes
#SBATCH --mem=0               # Request the full memory of the node
#SBATCH --time=23:0:0
#SBATCH --mail-user=gomeznor@ualberta.ca
#SBATCH --mail-type=ALL

cd ~/projects/def-bowling/diegog/laplacian_dual_dynamics
module load apptainer

parallel -j4 'CUDA_VISIBLE_DEVICES=$(({%} - 1)) apptainer exec --nv -B /home -B $PWD:/pwd -B /project -B /scratch -B /localscratch -B /project/def-mbowling/diegog/laplacian_dual_dynamics/ --pwd /pwd ~/apptainer/jax_cuda11.8.sif python train_laprepr.py --save_dir ~/logs/laplacian_dual_dynamics/ --config_file coefficient_augmented_martin.yaml --env_name {3} --seed {1} --regularization_weight {2} :::: ./rl_lap/hyperparam/seed_list_minimal_1-30.txt ::: 0.04 ::: GridRoom-64'

