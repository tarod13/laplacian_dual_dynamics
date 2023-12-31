#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=10
#SBATCH --mem-per-cpu=4G
#SBATCH --time=23:0:0
#SBATCH --mail-user=gomeznor@ualberta.ca
#SBATCH --mail-type=ALL

#SBATCH --array=0-3041

envs=("GridMaze-11" "GridRoom-64" "GridRoom-4" "GridRoomSym-4" "GridMaze-7" "GridMaze-17" "GridMaze-9" "GridMaze-32" "GridMaze-26" "GridRoom-32" "GridMaze-19" "GridRoom-16" "GridRoom-1")
bs=(100.0 50.0 20.0 10.0 5.0 2.0 1.0 0.5 0.2 0.1 0.05 0.02 0.01)
lrs=(0.0 0.001 0.01 0.1 1.0 10.0)
configs=("cqp.yaml" "sqp.yaml" "al.yaml")

N_SEED=$((${SLURM_ARRAY_TASK_ID} / 3042 + 1))
R_ENV=$((${SLURM_ARRAY_TASK_ID} % 3042))
N_ENV=$((${R_ENV} / 234))
R_B=$((${R_ENV} % 234))
N_B=$((${R_B} / 18))
R_LR=$((${R_B} % 18))
N_LR=$((${R_LR} / 3))
N_CONFIG=$((${R_LR} % 3))

SEEDS="./src/hyperparam/seed_list_minimal_${N_SEED}.txt"
CONFIG=${configs[$N_CONFIG]}
PROJECT_FOLDER="/project/def-mbowling/diegog/laplacian_dual_dynamics/"
SAVE_FOLDER="/scratch/"
ENV=${envs[$N_ENV]}
B=${bs[$N_B]}
LR=${lrs[$N_LR]}

cd ~/projects/def-mbowling/diegog/laplacian_dual_dynamics/
module load apptainer

parallel apptainer exec --nv -B /home -B $PWD:/pwd -B /project -W /scratch -B /localscratch -B $PROJECT_FOLDER --pwd /pwd ~/apptainer/lk_haiku_n.sif python3 train_laprepr.py "EXP1-YT" --use_wandb --wandb_offline --config_file $CONFIG --save_dir $SAVE_FOLDER --env_name {3} --seed {1} --barrier_initial_val {2} --lr_barrier_coefs {4} --n_samples 3000000 --total_train_steps 4000 :::: $SEEDS ::: $B ::: $ENV ::: $LR
