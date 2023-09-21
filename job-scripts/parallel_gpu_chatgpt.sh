#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=8G
#SBATCH --time=23:0:0
#SBATCH --mail-user=gomeznor@ualberta.ca
#SBATCH --mail-type=ALL

# Define your parameter sweep values
env_names=("GridMaze-19" "GridRoom-16")
reg_weights=(20.0 10.0 5.0 2.0 1.0 0.5 0.2 0.1 0.05 0.02 0.01)

# Set the number of tasks (number of parameter combinations)
num_tasks=$(( ${#env_names[@]} * ${#reg_weights[@]} ))

# Set up the environment
cd ~/projects/def-bowling/diegog/laplacian_dual_dynamics
module load apptainer

# Loop over parameter combinations and launch jobs
for env_name in "${env_names[@]}"; do
    for reg_weight in "${reg_weights[@]}"; do
        job_name="job_${env_name}_${reg_weight}"
        log_dir="~/logs/laplacian_dual_dynamics/${job_name}"

        # Submit a job for each parameter combination
        sbatch --export=ALL,env_name="$env_name",reg_weight="$reg_weight",log_dir="$log_dir" submit_job.sh
    done
done
