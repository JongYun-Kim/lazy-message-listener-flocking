#
import numpy as np
from numpy.typing import NDArray
from numpy import dtype
import torch
#
# Envs and models
from env.envs import LazyMsgListenersEnv
from model.lazy_listener import LazyListenerModelPPOTestMJ
#
# RLlib from Ray
import ray
from ray.rllib.policy.policy import Policy
from ray.rllib.models import ModelCatalog
from ray.rllib.policy.sample_batch import SampleBatch
from typing import Any, Dict, List, Type, Union
#
# Save files and load files
import pickle
import os  # creates dirs
from datetime import datetime  # gets current date and time
import time
#
# My utils
from utils.my_utils import compute_actions_and_probs


def compute_random_action(algo_name_, num_agents_):
    probabilities = {
        # "random_02": 0.2,
        "random_03": 0.3,
        "random_04": 0.4,
        "random_05": 0.5,
        "random_06": 0.6,
        "random_07": 0.7,
        "random_08": 0.8,
        "random_09": 0.9
    }

    if algo_name_ in probabilities:
        prob = probabilities[algo_name_]
        # Create the random action matrix
        action_ = np.random.choice([0, 1], size=(num_agents_, num_agents_), p=[1 - prob, prob]).astype(np.int8)
        np.fill_diagonal(action_, 1)
        # Set the action probabilities
        action_prob_ = np.full((num_agents_, num_agents_), prob, dtype=np.float32)
        np.fill_diagonal(action_prob_, 1.0)
    else:
        raise ValueError("In compute_random_action: Unknown algo_name: " + algo_name_)

    return action_, action_prob_


# def compute_metric_action(agent_positions, metric_distance):
#     # agent_positions: (num_agents, 2)
#     # metric_distance: float
#     # return: action_metric: (num_agents, num_agents)  # 1 if within metric_distance, 0 if not includes self
#     num_agents_met = agent_positions.shape[0]
#
#     # Get dummy action
#     action_metric = np.zeros((num_agents_met, num_agents_met), dtype=np.int8)
#
#     # Compute the relative distances between agents
#     rel_positions = agent_positions[:, np.newaxis, :] - agent_positions[np.newaxis, :, :]  # (num_agents, num_agents, 2)
#     rel_distances = np.linalg.norm(rel_positions, axis=-1)  # (num_agents, num_agents)
#     # Set the action to 1 for the agents within the metric distance
#     action_metric[rel_distances < metric_distance] = 1
#
#     return action_metric
#
#
# def compute_topology_action(agent_positions, num_neighbors):
#     # agent_positions: (num_agents, 2)
#     # num_neighbors: int
#     # return: action_topology: (num_agents, num_agents)  # 1 if neighbor, 0 if not includes self
#     num_agents_top = agent_positions.shape[0]
#
#     # Get dummy action
#     action_topology = np.zeros((num_agents_top, num_agents_top), dtype=np.int8)
#
#     # Compute the relative distances between agents
#     rel_positions = agent_positions[:, np.newaxis, :] - agent_positions[np.newaxis, :, :]  # (num_agents, num_agents, 2)
#     rel_distances = np.linalg.norm(rel_positions, axis=-1)  # (num_agents, num_agents)
#     # Sort the distances and get the indices
#     sorted_indices = np.argsort(rel_distances, axis=-1)
#     # Get the indices of the closest neighbors; include self
#     closest_neighbors = sorted_indices[:, :num_neighbors+1]  # +1 to include self
#     # Set the action to 1 for the closest neighbors
#     action_topology[np.arange(num_agents_top)[:, np.newaxis], closest_neighbors] = 1
#
#     return action_topology
#
#
# def get_metric_dict():
#     unit_distance = 25
#     met_dict = {}
#     for i in range(1, 10):
#         met_dict[f"metric_{i*10}"] = i * unit_distance  # i = 1, 2, ..., 9
#     return met_dict
#
#
# def get_topology_dict():
#     top_dict = {}
#     for i in range(1, 10):
#         top_dict[f"topology_{i*10}"] = 2 * i  # i = 1, 2, ..., 9
#     return top_dict


if __name__ == "__main__":
    # # do_debug = False
    # do_debug = True
    #
    # if do_debug:
    #     ray.init(local_mode=True)
    #
    # # Model settings
    # model_name = "lazy_listener_model_mj"
    # ModelCatalog.register_custom_model(model_name, LazyListenerModelPPOTestMJ)
    #
    # # Policy settings
    # base_path = "../../../ray_results/lazy_initial_test_030424"
    # trial_path = base_path + "/PPO_lazy_msg_listener_env_b33ce_00000_0_env_config=num_agents_pool_20_2024-03-07_10-30-21"
    # checkpoint_path = trial_path + "/checkpoint_000780/policies/default_policy"
    # policy = Policy.from_checkpoint(checkpoint_path)
    # policy.model.eval()

    # Experiment settings
    start_seed = 120
    num_seeds = 500
    num_algos = 7
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
    action_probs = np.zeros((num_seeds, num_algos, max_time_steps, num_agents, num_agents), dtype=np.float32)
    rewards = np.zeros((num_seeds, num_algos, max_time_steps), dtype=np.float32)
    control = np.zeros((num_seeds, num_algos, max_time_steps), dtype=np.float32)
    algo_str = ["random_03", "random_04", "random_05", "random_06", "random_07", "random_08", "random_09"]
    assert len(algo_str) == num_algos
    seeds = np.arange(num_seeds)
    seeds += start_seed  # start_seed to start_seed+num_seeds-1;  [10, ..., 19]

    start_time = time.time()
    # Seed loop: Experiments
    for i, seed in enumerate(seeds):
        for algo_index, algo_name in enumerate(algo_str):
            # Set seed and reset env
            env.seed(seed)
            obs = env.reset()
            done = False
            # Run upto the max time steps
            for t in range(max_time_steps):
                # Get action
                # if algo_name == "random_01":
                if algo_name.startswith("random_"):
                    action, action_prob = compute_random_action(algo_name, num_agents)
                else:
                    raise ValueError("Unknown algo_str: " + algo_name)
                # if algo_name == "ACS":  # TODO: You could implement these in a function
                #     action = np.ones((num_agents, num_agents), dtype=np.int8) if t == 0 else action
                #     action_prob = action
                # elif algo_name == "RL":
                #     action, action_prob = compute_actions_and_probs(policy, obs, num_agents, explore=True)
                #     action = action.astype(np.int8)
                # elif algo_name.startswith("metric"):
                #     metric_distance_ = metric_dict[algo_name]
                #     action = compute_metric_action(env.state["agent_states"][:, 0:2], metric_distance_)
                #     action_prob = action
                # elif algo_name.startswith("topology"):
                #     num_neighbors_ = topology_dict[algo_name]
                #     action = compute_topology_action(env.state["agent_states"][:, 0:2], num_neighbors_)
                #     action_prob = action
                # else:
                #     raise ValueError("Unknown algo_str: " + algo_name)
                # Step
                obs, reward, done, info = env.step(action)
                # Save
                trajectories[i, algo_index, t, :, :] = env.state["agent_states"][:, 0:2]
                velocities[i, algo_index, t, :, :] = env.state["agent_states"][:, 2:4]
                actions[i, algo_index, t, :, :] = action
                action_probs[i, algo_index, t, :, :] = action_prob
                rewards[i, algo_index, t] = reward
                control[i, algo_index, t] = reward + env.dt
                # Print progress
                if t % 10 == 0:
                    print(f"Progress: {i+1}/{num_seeds} seeds, {algo_name}: {t+1}/{max_time_steps} steps, "
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
        "actions": actions,            # (num_seeds, num_algos, max_time_steps, num_agents, num_agents)
        "action_probs": action_probs,  # (num_seeds, num_algos, max_time_steps, num_agents, num_agents)
        "rewards": rewards,  # (num_seeds, num_algos, max_time_steps)
        "control": control,  # (num_seeds, num_algos, max_time_steps)
        "algo_str": algo_str,  # (num_algos)
        "seeds": seeds,  # (num_seeds)
        "originated_from": "005_collect_random_data.py",
    }
    # Generate a timestamp for file naming
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # Save as a dictionary at ../data/today's date and time
    save_path = "../data/" + timestamp
    os.makedirs(save_path, exist_ok=True)  # exist_ok=True prevents error if directory already exists

    # Set filename using "001_acs_vs_rl_date_time"
    seed_range_str = f"seed_{seeds[0]}-{seeds[-1]}"
    file_name = "005_random_" + seed_range_str + "_" + timestamp + ".pkl"

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

