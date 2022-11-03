#!/bin/bash

# Slurm sbatch options
#SBATCH --job-name 10agent_sat_test
#SBATCH -a 0-1
#SBATCH --gres=gpu:volta:1
## SBATCH -n 10 # use with MPI # max cores request limit: -c 48 * 24; -n 48 * 24
#SBATCH -c 20 # cpus per task

# Loading the required module
source /etc/profile
module load anaconda/2022b

n_agents=10
logs_folder="out_Satelite_10fixed"
mkdir -p $logs_folder
# Run the script
# script to iterate through different hyperparameters
models="10_agent_informarl"
cent_obs="False"
seeds=(0 1)

# execute the script with different params
python -u onpolicy/scripts/train_mpe.py --use_valuenorm --use_popart \
--project_name "Satellite_Relative_Plots" \
--env_name "GraphMPE" \
--algorithm_name "rmappo" \
--seed "${seeds[$SLURM_ARRAY_TASK_ID]}" \
--experiment_name "${models}" \
--scenario_name "navigation_graph" \
--num_agents=${n_agents} \
--n_training_threads 1 --n_rollout_threads 128 \
--num_mini_batch 1 \
--episode_length 50 \
--num_env_steps 5000000 \
--ppo_epoch 10 --use_ReLU --gain 0.01 --lr 7e-4 --critic_lr 7e-4 \
--user_name "marl" \
--world_size 3 \
--use_cent_obs ${cent_obs} \
--graph_feat_type "relative" \
--model_dir "/home/gridsan/sdolan/Fair-MARL/10Agentlogs/" \
--auto_mini_batch_size --target_mini_batch_size 128 \
&> $logs_folder/out_${models}_${seeds[$SLURM_ARRAY_TASK_ID]}

#/home/gridsan/sdolan/Fair-MARL/informal-relative-centobsFalse/"

# python -u onpolicy/scripts/train_mpe.py --use_valuenorm --use_popart \
# --project_name "cent_obs_3" \
# --env_name "GraphMPE" \
# --algorithm_name "rmappo" \
# --seed 0 \
# --experiment_name "crap_bad" \
# --scenario_name "navigation_graph" \
# --num_agents=3 \
# --n_training_threads 1 --n_rollout_threads 2 \
# --num_mini_batch 1 \
# --episode_length 25 \
# --num_env_steps 2000 \
# --ppo_epoch 10 --use_ReLU --gain 0.01 --lr 7e-4 --critic_lr 7e-4 \
# --user_name "marl" \
# --use_cent_obs True \
# --auto_mini_batch_size --target_mini_batch_size 128 \
