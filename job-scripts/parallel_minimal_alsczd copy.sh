#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=10
#SBATCH --mem-per-cpu=4G
#SBATCH --time=23:0:0
#SBATCH --mail-user=gomeznor@ualberta.ca
#SBATCH --mail-type=ALL

#SBATCH --array=0-21

envs=("GridMaze-19" "GridRoom-16")
bs=(20.0 10.0 5.0 2.0 1.0 0.5 0.2 0.1 0.05 0.02 0.01)
configs=("augmented_sczd.yaml")

N_SEED=$((${SLURM_ARRAY_TASK_ID} / 22 + 1))
R_ENV=$((${SLURM_ARRAY_TASK_ID} % 22))
N_ENV=$((${R_ENV} / 11))
R_B=$((${R_ENV} % 11))
N_CONFIG=0
N_B=$((${R_B} / 1))

SEEDS="./rl_lap/hyperparam/seed_list_minimal_${N_SEED}.txt"
CONFIG=${configs[$N_CONFIG]}
PROJECT_FOLDER="/project/def-mbowling/diegog/laplacian_dual_dynamics/"
ENV=${envs[$N_ENV]}
B=${bs[$N_B]}

cd ~/projects/def-bowling/diegog/laplacian_dual_dynamics
module load apptainer

parallel apptainer exec --nv -B /home -B $PWD:/pwd -B /project -B /scratch -B /localscratch -B $PROJECT_FOLDER --pwd /pwd ~/apptainer/lk_haiku_n.sif python3 train_laprepr.py "E2" --use_wandb --save_dir ~/logs/laplacian_dual_dynamics/off --config_file $CONFIG --env_name {3} --seed {1} --barrier_initial_val {2} :::: $SEEDS ::: $B ::: $ENV
