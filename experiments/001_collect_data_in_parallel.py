#
# import numpy as np
#
# Envs and models
# from env.envs import LazyMsgListenersEnv, LazyMsgListenersTrainEnv
from model.lazy_listener import LazyListenerModelPPOTestMJ
#
# RLlib from Ray
import ray
# from ray.rllib.policy.policy import Policy
# from ray.rllib.models import ModelCatalog
#
# Save files and load files
import pickle
import os  # creates dirs
from datetime import datetime  # gets current date and time
import time


@ray.remote(num_cpus=1, num_gpus=0.5)  # Each task uses one CPU
def run_simulation(seed, algo_index, model_name, checkpoint_path, env_config, max_time_steps, num_agents, algo_str):
    from env.envs import LazyMsgListenersEnv  # Assuming this is the correct import path
    from ray.rllib.policy.policy import Policy
    from ray.rllib.models import ModelCatalog
    import numpy as np
    # import time

    # Register the model and load the policy
    ModelCatalog.register_custom_model(model_name, LazyListenerModelPPOTestMJ)
    policy = Policy.from_checkpoint(checkpoint_path)
    policy.config["num_gpus"] = 0.2
    policy.model.eval()

    # Initialize the environment
    env = LazyMsgListenersEnv(env_config)
    env.seed(seed)

    # Initialize result arrays for this simulation
    trajectories = np.zeros((max_time_steps, num_agents, 2), dtype=np.float32)
    velocities = np.zeros((max_time_steps, num_agents, 2), dtype=np.float32)
    spatial_entropy = np.zeros(max_time_steps, dtype=np.float32)
    velocity_entropy = np.zeros(max_time_steps, dtype=np.float32)
    actions = np.zeros((max_time_steps, num_agents, num_agents), dtype=np.int8)
    rewards = np.zeros(max_time_steps, dtype=np.float32)
    control = np.zeros(max_time_steps, dtype=np.float32)

    # Simulation loop
    obs = env.reset()
    for t in range(max_time_steps):
        # Get action based on the algorithm
        if algo_str[algo_index] == "ACS":
            action = np.ones((num_agents, num_agents), dtype=np.int64) if t == 0 else action
        elif algo_str[algo_index] == "RL stochastic":
            action = policy.compute_single_action(obs, explore=True)[0]
        elif algo_str[algo_index] == "RL deterministic":
            action = policy.compute_single_action(obs, explore=False)[0]
        else:
            raise ValueError("Unknown algo_str: " + algo_str[algo_index])

        # Environment step
        obs, reward, done, info = env.step(action)

        # Save results
        trajectories[t, :, :] = env.state["agent_states"][:, 0:2]
        velocities[t, :, :] = env.state["agent_states"][:, 2:4]
        actions[t, :, :] = action
        rewards[t] = reward
        control[t] = reward + env.dt  # Example computation, adjust as needed

        # Update entropy values if available
        # Example: spatial_entropy[t] = calculate_spatial_entropy(...)
        # Example: velocity_entropy[t] = calculate_velocity_entropy(...)

    return trajectories, velocities, spatial_entropy, velocity_entropy, actions, rewards, control


if __name__ == "__main__":
    # Initialize Ray
    ray.init(num_cpus=2, num_gpus=1)

    # Model and policy settings
    model_name = "lazy_listener_model_mj"
    base_path = "../../../ray_results/lazy_initial_test_030424"
    trial_path = base_path + "/PPO_lazy_msg_listener_env_b33ce_00000_0_env_config=num_agents_pool_20_2024-03-07_10-30-21"
    checkpoint_path = trial_path + "/checkpoint_000780/policies/default_policy"

    # Experiment settings
    start_seed = 20
    num_seeds = 14
    num_algos = 3
    num_agents = 20
    max_time_steps = 30
    env_config = {
        "num_agents_pool": [num_agents],
        "max_time_steps": max_time_steps,
    }
    algo_str = ["ACS", "RL stochastic", "RL deterministic"]

    # Seed array
    seeds = [start_seed + i for i in range(num_seeds)]

    # Dispatch Ray tasks
    tasks = []
    for seed in seeds:
        for algo_index in range(num_algos):
            task = run_simulation.remote(seed, algo_index, model_name, checkpoint_path, env_config, max_time_steps, num_agents, algo_str)
            tasks.append(task)

    # Collect results
    results = ray.get(tasks)

    # Process and save results
    # Here you can iterate through 'results' to extract and organize the data as needed.
    # For example, you could aggregate trajectories, velocities, actions, rewards, and control metrics across all seeds and algorithms.





    print("Done1")
    ray.shutdown()
    print("Done2")


