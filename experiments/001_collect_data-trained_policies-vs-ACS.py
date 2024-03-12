#
import numpy as np
#
# Envs and models
from env.envs import LazyMsgListenersEnv, LazyMsgListenersTrainEnv
from model.lazy_listener import LazyListenerModelPPOTestMJ
#
# RLlib from Ray
from ray.rllib.policy.policy import Policy
from ray.tune.registry import register_env
from ray.rllib.models import ModelCatalog
#
# Save files and load files
import pickle
import os  # creates dirs
from datetime import datetime  # gets current date and time
import time

if __name__ == "__main__":
    # Model settings
    model_name = "lazy_listener_model_mj"
    ModelCatalog.register_custom_model(model_name, LazyListenerModelPPOTestMJ)

    # Policy settings
    base_path = "../../../ray_results/lazy_initial_test_030424"
    trial_path = base_path + "/PPO_lazy_msg_listener_env_b33ce_00000_0_env_config=num_agents_pool_20_2024-03-07_10-30-21"
    checkpoint_path = trial_path + "/checkpoint_000780/policies/default_policy"
    policy = Policy.from_checkpoint(checkpoint_path)
    policy.model.eval()

    # Experiment settings
    start_seed = 10
    num_seeds = 10
    num_algos = 3
    num_agents = 20
    max_time_steps = 1000
    env_config = {
        "num_agents_pool": [num_agents],
        "max_time_steps": max_time_steps,
    }
    env = LazyMsgListenersEnv(env_config)  # seed it l8r

    # Result arrays
    trajectories = np.zeros((num_seeds, num_algos, max_time_steps, num_agents, 2), dtype=np.float32)
    velocities = np.zeros((num_seeds, num_algos, max_time_steps, num_agents, 2), dtype=np.float32)
    spatial_entropy = np.zeros((num_seeds, num_algos, max_time_steps), dtype=np.float32)
    velocity_entropy = np.zeros((num_seeds, num_algos, max_time_steps), dtype=np.float32)
    actions = np.zeros((num_seeds, num_algos, max_time_steps, num_agents, num_agents), dtype=np.int8)
    rewards = np.zeros((num_seeds, num_algos, max_time_steps), dtype=np.float32)
    control = np.zeros((num_seeds, num_algos, max_time_steps), dtype=np.float32)
    algo_str = ["ACS", "RL stochastic", "RL deterministic"]
    assert len(algo_str) == num_algos
    seeds = np.arange(num_seeds)
    seeds += start_seed  # start_seed to start_seed+num_seeds-1;  [10, ..., 19]

    start_time = time.time()
    # Seed loop: Experiments
    for i, seed in enumerate(seeds):
        for algo_index in range(num_algos):
            # Set seed and reset env
            env.seed(seed)
            obs = env.reset()
            done = False
            # Run upto the max time steps
            for t in range(max_time_steps):
                # Get action
                if algo_str[algo_index] == "ACS":  # TODO: You could implement these in a function
                    action = np.ones((num_agents, num_agents), dtype=np.int64) if t == 0 else action
                elif algo_str[algo_index] == "RL stochastic":
                    action = policy.compute_single_action(obs, explore=True)[0]
                elif algo_str[algo_index] == "RL deterministic":
                    action = policy.compute_single_action(obs, explore=False)[0]
                else:
                    raise ValueError("Unknown algo_str: " + algo_str[algo_index])
                # Step
                obs, reward, done, info = env.step(action)
                # Save
                trajectories[i, algo_index, t, :, :] = env.state["agent_states"][:, 0:2]
                velocities[i, algo_index, t, :, :] = env.state["agent_states"][:, 2:4]
                # spatial_entropy[i, algo_index, t] = env.std_pos_hist[t]
                # velocity_entropy[i, algo_index, t] = env.std_vel_hist[t]
                actions[i, algo_index, t, :, :] = action
                rewards[i, algo_index, t] = reward
                control[i, algo_index, t] = reward + env.dt
                # Print progress
                if algo_str[algo_index] != "ACS":
                    print(f"Progress: {i+1}/{num_seeds} seeds, {algo_str[algo_index]}: {t+1}/{max_time_steps} steps, "
                          f"took {time.time()-start_time:.2f} seconds")
            # Batch save spatial and velocity entropy for efficiency
            spatial_entropy[i, algo_index, :] = env.std_pos_hist
            velocity_entropy[i, algo_index, :] = env.std_vel_hist

    # Make the arrays into a dictionary
    data = {
        "trajectories": trajectories,  # (num_seeds, num_algos, max_time_steps, num_agents, 2)
        "velocities": velocities,      # (num_seeds, num_algos, max_time_steps, num_agents, 2)
        "spatial_entropy": spatial_entropy,    # (num_seeds, num_algos, max_time_steps)
        "velocity_entropy": velocity_entropy,  # (num_seeds, num_algos, max_time_steps)
        "actions": actions,  # (num_seeds, num_algos, max_time_steps, num_agents, num_agents)
        "rewards": rewards,  # (num_seeds, num_algos, max_time_steps)
        "control": control,  # (num_seeds, num_algos, max_time_steps)
        "algo_str": algo_str,  # (num_algos)
        "seeds": seeds,  # (num_seeds)
        "originated_from": "001_collect_data-trained_policies-vs-ACS.py",
    }
    # Generate a timestamp for file naming
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # Save as a dictionary at ../data/today's date and time
    save_path = "../data/" + timestamp
    os.makedirs(save_path, exist_ok=True)  # exist_ok=True prevents error if directory already exists

    # Set filename using "001_acs_vs_rl_date_time"
    seed_range_str = f"seed_{seeds[0]}-{seeds[-1]}"
    file_name = "001_acs_vs_rl_" + seed_range_str + "_" + timestamp + ".pkl"

    # Save
    with open(os.path.join(save_path, file_name), "wb") as f:
        pickle.dump(data, f)

    print("Data saved at: " + os.path.join(save_path, file_name))
    print(f"Current time: {timestamp}")
    # Calculate elapsed time in a human-readable format (days, hours, min, sec)
    elapsed_time = time.time() - start_time
    elapsed_time = time.gmtime(elapsed_time)
    print(f"Elapsed time: {elapsed_time.tm_mday-1} days, {elapsed_time.tm_hour} hours, {elapsed_time.tm_min} minutes, "
          f"{elapsed_time.tm_sec} seconds")
