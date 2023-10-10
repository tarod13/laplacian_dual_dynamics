#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=4G
#SBATCH --time=23:0:0
#SBATCH --mail-user=gomeznor@ualberta.ca
#SBATCH --mail-type=ALL

seeds=(1 2 3 4 5 6)
envs=("GridMaze-19" "GridRoom-16" "GridRoom-1")
bs=(20.0 10.0 5.0 2.0 1.0 0.5 0.2 0.1 0.05 0.02 0.01)
lrs=(0.0 0.001 0.01 0.1 1.0)
configs=("cqp.yaml" "sqp.yaml" "al.yaml")

# Set the number of tasks (number of parameter combinations)
num_tasks=$(( ${#seeds[@]} * ${#envs[@]} * ${#bs[@]} * ${#lrs[@]} * ${#configs[@]} ))

# Set up the environment
cd ~/projects/def-mbowling/diegog/laplacian_dual_dynamics/


# Loop over parameter combinations and launch jobs
for seed in "${seeds[@]}"; do
    SEEDS="./src/hyperparam/seed_list_minimal_${seed}.txt"
    for env in "${envs[@]}"; do
        for b in "${bs[@]}"; do
            for lr in "${lrs[@]}"; do
                for config in "${configs[@]}"; do
                    # Submit a job for each parameter combination
                    sbatch --export=ALL,SEED="$SEEDS",ENV="$env",B=$b,LR=$lr,CONFIG="$config" job-scripts/exp1_min_single.sh
                done
            done
        done
    done
done
