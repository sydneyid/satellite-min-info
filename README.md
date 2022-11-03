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

## Experiment: Optimal Minibatch:
* git checkout sid/speed
* In compare7.sh:
    * num_training_threads = 4
    * n_rollout_threads = 2
    * num_mini_batch = 1
    * episode_lengths=(25n), we vary n
    * num_env_steps = episode_lengths * n_rollout_threads
    * Then mini_batch_size = n * n_rollout_threads/2 * 15/num_mini_batch
* Run with different values of n and record Training steps per second for each mini_batch_size
* Results: https://www.desmos.com/calculator/b97lzbau8q
    * Optimal mini_batch_size ~= 250

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
