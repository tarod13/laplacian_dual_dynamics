#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=10
#SBATCH --mem-per-cpu=4G
#SBATCH --time=23:0:0
#SBATCH --mail-user=gomeznor@ualberta.ca
#SBATCH --mail-type=ALL

SEEDS=$1
CONFIG=$2
PROJECT_FOLDER="/project/def-mbowling/diegog/laplacian_dual_dynamics/"
ENV=$3
B=$4
LR=$5

cd ~/projects/def-mbowling/diegog/laplacian_dual_dynamics/
module load apptainer

parallel apptainer exec --nv -B /home -B $PWD:/pwd -B /project -B /scratch -B /localscratch -B $PROJECT_FOLDER --pwd /pwd ~/apptainer/lk_haiku_n.sif python3 train_laprepr.py "EXP1-M" --use_wandb --wandb_offline --config_file $CONFIG --env_name "$ENV" --seed {1} --barrier_initial_val $B --lr_barrier_coefs $LR :::: $SEEDS