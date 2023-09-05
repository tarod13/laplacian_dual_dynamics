#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=10
#SBATCH --mem-per-cpu=8G
#SBATCH --time=23:0:0
#SBATCH --mail-user=gomeznor@ualberta.ca
#SBATCH --mail-type=ALL

#SBATCH --array=0-7

envs=("GridRoom-64" "GridRoom-16" "GridMaze-19" "GridRoom-1" "GridRoom-4" "GridRoomSym-4" "GridMaze-7" "GridMaze-17")
hiddens=("256 256")
configs=("dual_exact.yaml")

N_SEED=$((${SLURM_ARRAY_TASK_ID} / 8 + 4))
R_ENV=$((${SLURM_ARRAY_TASK_ID} % 8))
N_ENV=$((${R_ENV} / 1))
N_CONFIG=0
N_HIDDEN=0

SEEDS="./rl_lap/hyperparam/seed_list_minimal_${N_SEED}.txt"
CONFIG=${configs[$N_CONFIG]}
HIDDEN=${hiddens[$N_HIDDEN]}
PROJECT_FOLDER="/project/def-mbowling/diegog/laplacian_dual_dynamics/"
ENV=${envs[$N_ENV]}

cd ~/projects/def-bowling/diegog/laplacian_dual_dynamics
module load apptainer

parallel apptainer exec --nv -B /home -B $PWD:/pwd -B /project -B /scratch -B /localscratch -B $PROJECT_FOLDER --pwd /pwd ~/apptainer/lk_haiku_n.sif python3 train_laprepr.py --save_dir ~/logs/laplacian_dual_dynamics/off --config_file $CONFIG --env_name {2} --seed {1} --total_train_steps {3} --max_episode_steps {4} --lr {5} --hidden_dims $HIDDEN --batch_size {6} --discount {7} --n_samples {8} :::: $SEEDS ::: $ENV ::: 8000000 ::: 50 ::: 0.0001 ::: 32 ::: 0.9 ::: 3000000
#apptainer exec --nv -B /home -B $PWD:/pwd -B /project -B /scratch -B /localscratch -B /project/def-mbowling/diegog/laplacian_dual_dynamics/ --pwd /pwd ~/apptainer/lk_haiku.sif python3 train_laprepr.py --save_dir ~/logs/laplacian_dual_dynamics/off --config_file coefficient_augmented_martin.yaml --env_name GridRoom-1 --seed 1234 --regularization_weight 5.0 --total_train_steps 400000 --max_episode_steps 1000000 --lr 0.0001 --hidden_dims "256 256" --batch_size 32 --discount 0.9 --n_samples 2000000
#apptainer shell --nv -B /home -B $PWD:/pwd -B /project -B /scratch -B /localscratch -B /project/def-mbowling/diegog/laplacian_dual_dynamics/ --pwd /pwd ~/apptainer/lk_haiku.sif
