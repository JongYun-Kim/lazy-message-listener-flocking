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


def compute_fn_action(agent_positions, num_neighbors):
    # agent_positions: (num_agents, 2)
    # num_neighbors: int
    # return: action_fn: (num_agents, num_agents)  # 1 if neighbor, 0 if not includes self
    num_agents_top = agent_positions.shape[0]

    # Get dummy action
    action_FN= np.zeros((num_agents_top, num_agents_top), dtype=np.int8)

    # Compute the relative distances between agents
    rel_positions = agent_positions[:, np.newaxis, :] - agent_positions[np.newaxis, :, :]  # (num_agents, num_agents, 2)
    rel_distances = np.linalg.norm(rel_positions, axis=-1)  # (num_agents, num_agents)
    # Sort the distances and get the indices
    sorted_indices = np.argsort(-rel_distances, axis=-1)
    # Get the indices of the farthest neighbors; include self
    farthest_neighbors = sorted_indices[:, :num_neighbors]  # not including self for now
    # Set the action to 1 for the closest neighbors
    action_FN[np.arange(num_agents_top)[:, np.newaxis], farthest_neighbors] = 1
    # Ensure self-loops are included
    np.fill_diagonal(action_FN, 1)

    return action_FN


def compute_as_action(agent_positions, num_sections):
    num_agents_ = agent_positions.shape[0]

    # Initialize the adjacency matrix
    action_AS = np.zeros((num_agents_, num_agents_), dtype=np.int8)

    # Calculate the angles for each section
    angles = np.linspace(0, 2 * np.pi, num_sections + 1)

    for i in range(num_agents_):
        # Compute the relative positions and distances from agent i to all other agents
        rel_positions = agent_positions - agent_positions[i]  # (num_agents, 2)
        rel_distances = np.linalg.norm(rel_positions, axis=-1)  # (num_agents,)

        # Compute the angles of these relative positions
        rel_angles = np.arctan2(rel_positions[:, 1], rel_positions[:, 0]) % (2 * np.pi)  # (num_agents,)

        # Iterate over each section
        for j in range(num_sections):
            # Get the indices of agents in the current angular section
            indices_in_section = np.where((rel_angles >= angles[j]) & (rel_angles < angles[j + 1]))[0]

            if len(indices_in_section) > 0:
                # Find the closest agent in this section, excluding the current agent itself
                valid_indices = indices_in_section[indices_in_section != i]
                if len(valid_indices) > 0:
                    closest_agent_idx = valid_indices[np.argmin(rel_distances[valid_indices])]
                    action_AS[i, closest_agent_idx] = 1

    # Ensure self-loops are included
    np.fill_diagonal(action_AS, 1)

    return action_AS


def compute_va_action(agent_positions, agent_velocities, selection_portion):
    num_agents__ = agent_positions.shape[0]

    # Initialize the adjacency matrix
    action_va = np.zeros((num_agents__, num_agents__), dtype=np.int8)

    # Iterate over each agent
    for i in range(num_agents__):
        # Compute the relative positions from agent i to all other agents
        rel_positions = agent_positions - agent_positions[i]  # (num_agents, 2)
        rel_distances = np.linalg.norm(rel_positions, axis=-1)  # (num_agents,)

        # Compute the heading angle of agent i
        heading_angle = np.arctan2(agent_velocities[i, 1], agent_velocities[i, 0])  # Scalar

        # Compute the angles of these relative positions with respect to the heading angle
        rel_angles = (np.arctan2(rel_positions[:, 1], rel_positions[:, 0]) - heading_angle) % (2 * np.pi)  # (num_agents,)

        # Define the visible area
        lower_bound = (-0.6 * np.pi) % (2 * np.pi)
        upper_bound = (0.6 * np.pi) % (2 * np.pi)

        # Get the indices of agents in the visible area
        if lower_bound < upper_bound:
            indices_in_area = np.where((rel_angles >= lower_bound) & (rel_angles <= upper_bound))[0]
        else:  # Wraps around the 0 angle
            indices_in_area = np.where((rel_angles >= lower_bound) | (rel_angles <= upper_bound))[0]

        # Calculate the number of agents to select based on selection_portion
        # num_to_select = int(np.ceil(len(indices_in_area) * selection_portion))
        num_to_select = int(np.floor(len(indices_in_area) * selection_portion))

        if num_to_select > 0:
            # Find the farthest agents in the visible area
            farthest_indices = indices_in_area[np.argsort(-rel_distances[indices_in_area])[:num_to_select]]
            action_va[i, farthest_indices] = 1

    # Ensure self-loops are included
    np.fill_diagonal(action_va, 1)

    return action_va


if __name__ == "__main__":
    # # do_debug = False
    # do_debug = True
    #
    # if do_debug:
    #     ray.init(local_mode=True)
    # #
    # # Model settings
    # model_name = "lazy_listener_model_mj"
    # ModelCatalog.register_custom_model(model_name, LazyListenerModelPPOTestMJ)
    # #
    # # Policy settings
    # base_path = "../../../ray_results/lazy_initial_test_030424"
    # trial_path = base_path + "/PPO_lazy_msg_listener_env_b33ce_00000_0_env_config=num_agents_pool_20_2024-03-07_10-30-21"
    # checkpoint_path = trial_path + "/checkpoint_000780/policies/default_policy"
    # policy = Policy.from_checkpoint(checkpoint_path)
    # policy.model.eval()

    # Experiment settings
    start_seed = 120
    num_seeds = 200
    num_algos = 9
    num_agents = 20
    max_time_steps = 1000
    env_config = {
        "num_agents_pool": [num_agents],
        "max_time_steps": max_time_steps,
    }
    env = LazyMsgListenersEnv(env_config)  # seed it l8r

    # fd1 = "./../data/2024-03-16_16-08-55/"
    # fn1 = "002_acs_rl_nature_seed_120-619_2024-03-16_16-08-55.pkl"
    # with open(fd1 + fn1, "rb") as f:
    #     data1 = pickle.load(f)

    # Result arrays
    trajectories = np.zeros((num_seeds, num_algos, max_time_steps, num_agents, 2), dtype=np.float32)
    velocities = np.zeros((num_seeds, num_algos, max_time_steps, num_agents, 2), dtype=np.float32)
    spatial_entropy = np.zeros((num_seeds, num_algos, max_time_steps), dtype=np.float32)
    velocity_entropy = np.zeros((num_seeds, num_algos, max_time_steps), dtype=np.float32)
    actions = np.zeros((num_seeds, num_algos, max_time_steps, num_agents, num_agents), dtype=np.int8)
    action_probs = np.zeros((num_seeds, num_algos, max_time_steps, num_agents, num_agents), dtype=np.float32)
    rewards = np.zeros((num_seeds, num_algos, max_time_steps), dtype=np.float32)
    control = np.zeros((num_seeds, num_algos, max_time_steps), dtype=np.float32)
    # algo_str = ["ACS", "RL", "FN_1", "FN_2", "FN_3", "AS_1", "AS_2", "AS_3", "VA_1", "VA_2", "VA_3"]
    algo_str = ["FN_1", "FN_2", "FN_3", "AS_1", "AS_2", "AS_3", "VA_1", "VA_2", "VA_3"]
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
                if algo_name == "ACS":  # TODO: You could implement these in a function
                    action = np.ones((num_agents, num_agents), dtype=np.int8) if t == 0 else action
                    action_prob = action
                # elif algo_name == "RL":
                #     action, action_prob = compute_actions_and_probs(policy, obs, num_agents, explore=True)
                #     action = action.astype(np.int8)
                elif algo_name.startswith("FN"):
                    num_fa = 6*(int(algo_name[-1]))
                    action = compute_fn_action(env.state["agent_states"][:, 0:2], num_fa)
                    action_prob = action
                elif algo_name.startswith("AS"):
                    num_sec = 4*(int(algo_name[-1]))
                    action = compute_as_action(env.state["agent_states"][:, 0:2], num_sec)
                    action_prob = action
                elif algo_name.startswith("VA"):
                    portion = 0.3*(int(algo_name[-1]))
                    action = compute_va_action(env.state["agent_states"][:, 0:2], env.state["agent_states"][:, 2:4], portion)
                    action_prob = action
                else:
                    raise ValueError("Unknown algo_str: " + algo_name)
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
        "originated_from": "006_collect_heuristic_data.py",
    }
    # Generate a timestamp for file naming
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # Save as a dictionary at ../data/today's date and time
    save_path = "../data/" + timestamp
    os.makedirs(save_path, exist_ok=True)  # exist_ok=True prevents error if directory already exists

    # Set filename using "001_acs_vs_rl_date_time"
    seed_range_str = f"seed_{seeds[0]}-{seeds[-1]}"
    file_name = "006_heu_" + seed_range_str + "_" + timestamp + ".pkl"

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

