#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=4G
#SBATCH --time=23:0:0
#SBATCH --mail-user=gomeznor@ualberta.ca
#SBATCH --mail-type=ALL

#SBATCH --array=0-779

envs=("GridRoom-64" "GridMaze-11" "GridRoom-4" "GridRoomSym-4" "GridMaze-7" "GridMaze-17" "GridMaze-32" "GridMaze-26" "GridRoom-32" "GridMaze-9" "GridMaze-19" "GridRoom-1" "GridRoom-16")
# 11 enough ('N', 'Y', 'Y', 'Y', 'Y', 'Y', 'Y', 'Y', 'Y', 'Y', 'Y', 'Y', 'Y')
# 12 enough ('Y', 'Y', 'Y', 'Y', 'N', 'Y', 'Y', 'Y', 'Y', 'Y', 'Y', 'N', 'Y')
bs=(2)
lrs=(0)
configs=("cqp.yaml")

N_SEED=$((${SLURM_ARRAY_TASK_ID} / 13 + 1))
R_ENV=$((${SLURM_ARRAY_TASK_ID} % 13))
N_ENV=$((${R_ENV} / 1))
R_B=$((${R_ENV} % 1))
N_B=$((${R_B} / 1))
R_LR=$((${R_B} % 1))
N_LR=$((${R_LR} / 1))
N_CONFIG=$((${R_LR} % 1))

SEED_FILE="./src/hyperparam/seed_list.txt"
SEED=$(sed -n "${N_SEED}p" "$SEED_FILE")
CONFIG=${configs[$N_CONFIG]}
PROJECT_FOLDER="/project/def-mbowling/diegog/laplacian_dual_dynamics/"
ENV=${envs[$N_ENV]}
B=${bs[$N_B]}
LR=${lrs[$N_LR]}

cd ~/projects/def-mbowling/diegog/laplacian_dual_dynamics/
module load apptainer

apptainer exec --nv -B /home -B $PWD:/pwd -B /project -W /scratch -B /localscratch -B $PROJECT_FOLDER --pwd /pwd ~/apptainer/lk_haiku_n.sif python3 train_laprepr.py "EXP-PERM-0" --use_wandb --config_file $CONFIG --obs_mode "xy" --env_name $ENV --seed $SEED --barrier_initial_val $B --lr_barrier_coefs $LR --n_samples 1000000 --total_train_steps 200000