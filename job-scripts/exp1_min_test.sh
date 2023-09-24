#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=10
#SBATCH --mem-per-cpu=4G
#SBATCH --time=23:0:0
#SBATCH --mail-user=gomeznor@ualberta.ca
#SBATCH --mail-type=ALL

envs=("GridMaze-19")
bs=(1.0)
lrs=(0.0)
configs=("cqp.yaml")

N_ENV=0
N_B=0
N_LR=0
N_CONFIG=0

SEEDS="./src/hyperparam/seed_list_min.txt"
CONFIG=${configs[$N_CONFIG]}
PROJECT_FOLDER="/project/def-mbowling/diegog/laplacian_dual_dynamics/"
ENV=${envs[$N_ENV]}
B=${bs[$N_B]}
LR=${lrs[$N_LR]}

cd ~/projects/def-mbowling/diegog/laplacian_dual_dynamics/
module load apptainer

parallel apptainer exec --nv -B /home -B $PWD:/pwd -B /project -B /scratch -B /localscratch -B $PROJECT_FOLDER --pwd /pwd ~/apptainer/lk_haiku_n.sif python3 train_laprepr.py "EXP1-T" --use_wandb --wandb_offline --config_file $CONFIG --env_name {3} --seed {1} --barrier_initial_val {2} --lr_barrier_coefs {4} :::: $SEEDS ::: $B ::: $ENV ::: $LR
