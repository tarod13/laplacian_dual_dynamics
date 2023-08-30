#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=10
#SBATCH --mem-per-cpu=8G
#SBATCH --time=23:0:0
#SBATCH --mail-user=gomeznor@ualberta.ca
#SBATCH --mail-type=ALL

#SBATCH --array=0-149

envs=("GridRoom-1" "GridRoom-4" "GridRoom-16" "GridMaze-19" "GridRoom-64")
ws=(5.0 5.0 1.0 1.0 0.04)
hiddens=("256 256")
batches=(32)
discounts=(0.0 0.25 0.5 0.75 0.9 0.95 0.975 0.99 0.995 0.999)

N_SEED=$((${SLURM_ARRAY_TASK_ID} / 50 +1))
R_ENV=$((${SLURM_ARRAY_TASK_ID} % 50))
N_ENV=$((${R_ENV} / 10))
N_HIDDEN=0
N_BATCH=0
N_DISC=$((${R_ENV} % 10))

SEEDS="./rl_lap/hyperparam/seed_list_minimal_${N_SEED}.txt"
CONFIG="coefficient_augmented_martin.yaml"
HIDDEN=${hiddens[$N_HIDDEN]}
PROJECT_FOLDER="/project/def-mbowling/diegog/laplacian_dual_dynamics/"
ENV=${envs[$N_ENV]}
W=${ws[$N_ENV]}
BATCH_SIZE=${batches[$N_BATCH]}
DISCOUNT=${discounts[$N_DISC]}

cd ~/projects/def-bowling/diegog/laplacian_dual_dynamics
module load apptainer

parallel apptainer exec --nv -B /home -B $PWD:/pwd -B /project -B /scratch -B /localscratch -B $PROJECT_FOLDER --pwd /pwd ~/apptainer/lk_haiku.sif python3 train_laprepr.py --save_dir ~/logs/laplacian_dual_dynamics/off --config_file $CONFIG --env_name {3} --seed {1} --regularization_weight {2} --total_train_steps {4} --max_episode_steps {5} --lr {6} --hidden_dims $HIDDEN --batch_size {7} --discount {8} --n_samples {9} :::: $SEEDS ::: $W ::: $ENV ::: 1000000 ::: 50 ::: 0.0001 ::: $BATCH_SIZE ::: $DISCOUNT ::: 2000000
