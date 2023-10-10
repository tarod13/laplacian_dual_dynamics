#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=10
#SBATCH --mem-per-cpu=4G
#SBATCH --time=23:0:0
#SBATCH --mail-user=gomeznor@ualberta.ca
#SBATCH --mail-type=ALL

#SBATCH --array=0-143

envs=("GridMaze-19" "GridRoom-16" "GridRoom-1" "GridRoom-64" "GridRoom-4" "GridRoomSym-4" "GridMaze-7" "GridMaze-17" "GridMaze-9" "GridMaze-32" "GridMaze-26" "GridRoom-32")
bs=(2.0 0.5)
lrs=(0.0 0.01)
configs=("cqp.yaml" "al.yaml")

N_SEED=$((${SLURM_ARRAY_TASK_ID} / 24 + 1))
R_ENV=$((${SLURM_ARRAY_TASK_ID} % 24))
N_ENV=$((${R_ENV} / 2))
N_CONFIG=$((${R_ENV} % 2))


SEEDS="./src/hyperparam/seed_list_minimal_${N_SEED}.txt"
CONFIG=${configs[$N_CONFIG]}
PROJECT_FOLDER="/project/def-mbowling/diegog/laplacian_dual_dynamics/"
ENV=${envs[$N_ENV]}
B=${bs[$N_CONFIG]}
LR=${lrs[$N_CONFIG]}

cd ~/projects/def-mbowling/diegog/laplacian_dual_dynamics/
module load apptainer

parallel apptainer exec --nv -B /home -B $PWD:/pwd -B /project -B /scratch -B /localscratch -B $PROJECT_FOLDER --pwd /pwd ~/apptainer/lk_haiku_n.sif python3 train_laprepr.py "EXP2" --use_wandb --wandb_offline --config_file $CONFIG --env_name {3} --seed {1} --barrier_initial_val {2} --lr_barrier_coefs {4} --n_samples 3000000 --total_train_steps 200000 :::: $SEEDS ::: $B ::: $ENV ::: $LR
