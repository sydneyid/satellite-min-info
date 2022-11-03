# satellite-min-info
Using graph neural networks for model the minimum amount of information needed for successful collision avoidance in space traffic management. 


## Dependencies:
* [Multiagent-particle-envs](https://github.com/openai/multiagent-particle-envs): We have pulled the relevant folder from the repo to modify it.
    * `pip install gym==0.10.5` (newer versions also seem to work)
    * `pip install numpy-stl`
    * torch==1.8.0              
    * torch-geometric==2.0.3
    * torch-scatter==2.0.8
    * torch-sparse==0.6.12
    * wandb
   

## Syntax to Train the File
The .sh files are used to run batchs files to train this code. The base .sh file to run is sat_train.sh. If you would like to run the code from the terminal, the syntax is as follows:

`python -u onpolicy/scripts/train_mpe.py --use_valuenorm --use_popart --project_name "cent_obs_3"  --env_name "GraphMPE" --algorithm_name "rmappo" --seed 0 --experiment_name "crap_bad" --scenario_name "navigation_graph" --num_agents=3 --n_training_threads 1 --n_rollout_threads 2 --num_mini_batch 1  --episode_length 25 --num_env_steps 2000  --ppo_epoch 10 --use_ReLU --gain 0.01 --lr 7e-4 --critic_lr 7e-4 --user_name "marl"  --use_cent_obs True --auto_mini_batch_size --target_mini_batch_size 128`
 
scenario_name refers to the scenario type that you want the problem to be tested on. Navigation_graph is for navigating with a GNN, whereas navigation is for navigation with standard MARL. 

experiment_name is a custom name to be able to find that set of experiments in wandb later

algorithm_name is the type of algorithm that you want to test in the space environment.


The syntax to train the file on one of the comparative baseline algorithms is:
`python -u onpolicy/scripts/train_mpe.py –use_valuenorm –use_popart –project_name “crap_bad” –env_name “MPE” –algorithm_name “rmappo” –seed 0 –experiment_name “crap_bad” –scenario_name “navigation” –num_agents=3 –n_rollout_threads 2 –episode length 25 –tau 0.005 –num_env_steps 2000 –batch_size 10 –lr 7e-4 –use_name “marl” –use_cent_obs False –auto_mini_batch_size –target_mini_batch_size 128`

env_name is used to configure the environment for the regular multi agent case. Similarly, the scenario is for navigating without a GNN.


 

## Syntax to Test the File
Run from the root folder. You need to have a file directory, which contains a config.yaml file, an actor.pt file, and a critic.pt file. in this example, these files are stored at '/Users/sdolan/test_files'. For a full list of parameters that you can change when evaluating the model, look at the eval_mpe.py file.

'python onpolicy/scripts/eval_mpe.py --model_dir='/Users/sdolan/test_files' --render_episodes=2 --world_size=2 --num_agents=3 --num_obstacles=3' 



## Troubleshooting:
* `OMP: Error #15: Initializing libiomp5.dylib, but found libomp.dylib already initialized.`: Install nomkl by running [`conda install nomkl`](https://stackoverflow.com/questions/53014306/error-15-initializing-libiomp5-dylib-but-found-libiomp5-dylib-already-initial)

* `AttributeError: dlsym(RTLD_DEFAULT, CFStringCreateWithCString): symbol not found`: This issue arises with MacOS Big Sur. A hacky fix for this is to revert change the `pyglet` version to maintenance version using `pip install --user --upgrade git+http://github.com/pyglet/pyglet@pyglet-1.5-maintenance`

* `AttributeError: 'NoneType' object has no attribute 'origin'`: This error arises whilst using `torch-geometric` with CUDA. Uninstall `torch_geometric`, `torch-cluster`, `torch-scatter`, `torch-sparse`, and `torch-spline-conv`. Then re-install using:
    ```
    TORCH="1.8.0"
    CUDA="cu102"
    pip install --no-index torch-scatter -f https://data.pyg.org/whl/torch-${TORCH}+${CUDA}.html --user
    pip install --no-index torch-sparse -f https://data.pyg.org/whl/torch-${TORCH}+${CUDA}.html --user
    pip install torch-geometric --user
    ```
