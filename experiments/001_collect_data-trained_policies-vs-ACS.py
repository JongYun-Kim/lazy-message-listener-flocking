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


def batch_observations(
        obs_list: List[Dict[str, np.ndarray]],
        get_input_dict: bool = True,
        use_torch: bool = False,
) -> Dict[str, np.ndarray]:
    """
    Batch a list of single observations into a format suitable for RLlib's compute_actions_from_input_dict method.
    Args:
        obs_list (List[Dict[str, np.ndarray]]): A list where each element is a single observation from the environment.
        get_input_dict (bool): A flag indicating whether to return the batched observations as a SampleBatch object.
        use_torch (bool): A flag indicating whether to use PyTorch tensors or numpy arrays for the batched observations.
    Returns:

    Use something like this:
    batch_size = 5

    obs_list = []
    env_list = []
    for seed in range(batch_size):
        env = LazyMsgListenersEnv(config)
        env.seed(seed)
        obs = env.reset()
        obs_list.append(obs)
        env_list.append(env)
    input_dict = batch_observations(obs_list, get_input_dict=True)
    actions = policy.compute_actions_from_input_dict(input_dict=input_dict, explore=False)[0]
    """
    batched_obs = {}

    # Iterate over each observation and batch them together
    for obs in obs_list:
        for key, value in obs.items():
            if key not in batched_obs:
                batched_obs[key] = []
            batched_obs[key].append(value)

    # Convert lists to tensors or appropriately shaped numpy arrays
    for key, value in batched_obs.items():
        if use_torch:  # Using PyTorch tensors
            batched_obs[key] = torch.tensor(value)
        else:  # Using numpy arrays
            batched_obs[key] = np.array(value)

    if get_input_dict:
        return {SampleBatch.OBS: SampleBatch(batched_obs)}
    else:
        return batched_obs


def softmax(input, dim=None):
    """
    Applies a softmax function using NumPy.
    Args:
        input (numpy.ndarray): input array
        dim (int): A dimension along which softmax will be computed.
    Returns:
        numpy.ndarray: softmax applied array
    """
    # if dim is None, we assume the softmax should be applied to the last dimension
    if dim is None:
        dim = -1

    # Shift input for numerical stability
    input_shifted = input - np.max(input, axis=dim, keepdims=True)
    # Calculate the exponential of the input
    exp_input = np.exp(input_shifted)
    # Sum of exponentials along the specified dimension
    sum_exp_input = np.sum(exp_input, axis=dim, keepdims=True)
    # Compute the softmax
    softmax_output = exp_input / sum_exp_input

    return softmax_output


def compute_actions_and_probs(
        policy: Policy,
        obs: Dict[str, NDArray[dtype]],
        num_agents_: int,
        explore: bool = True,
        batch_mode: bool = False,
):
    """
    Compute actions using the given policy.
    Args:
        policy (Policy): The policy to use for computing the actions.
        obs: dict
        explore (bool): Whether to use exploration when computing the actions.
    Returns:
    """
    if batch_mode:
        assert isinstance(obs, list), "When batch_mode is True, obs must be a list of observations."
        batch_size = len(obs)
        input_dict = batch_observations(obs, get_input_dict=True)
        action_info = policy.compute_actions_from_input_dict(input_dict=input_dict, explore=explore)
        action_ = action_info[0]  # (batch_size, num_agents, num_agents)
        logits = action_info[2]['action_dist_inputs']  # flattened logits (batch_size, num_agents * num_agents * 2)
        logits = logits.reshape(batch_size, num_agents_, num_agents_, 2)  # (batch_size, num_agents, num_agents, 2)
        action_probs_ = softmax(logits, dim=-1)[:, :, :, 1]  # (batch_size, num_agents, num_agents)
    else:
        # Compute the action
        action_info = policy.compute_single_action(obs, explore=explore)
        action_ = action_info[0]
        logits = action_info[2]['action_dist_inputs']  # flattened logits (batch_size, num_agents * num_agents * 2)
        logits = logits.reshape(num_agents_, num_agents_, 2)  # (num_agents, num_agents, 2)
        action_probs_ = softmax(logits)[:, :, 1]  # (num_agents, num_agents)

    return action_, action_probs_


def compute_metric_action():
    pass


def compute_topology_action():
    pass


if __name__ == "__main__":
    # do_debug = False
    do_debug = True

    if do_debug:
        import ray
        ray.init(local_mode=True)

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
    start_seed = 20
    num_seeds = 100
    num_algos = 2
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
    algo_str = ["ACS", "RL stochastic"]
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
                    action_prob = action
                elif algo_str[algo_index] == "RL stochastic":
                    action, action_prob = compute_actions_and_probs(policy, obs, num_agents, explore=True)
                    # action_tuple = policy.compute_single_action(obs, explore=True)
                    # action = action_tuple[0]
                else:
                    raise ValueError("Unknown algo_str: " + algo_str[algo_index])
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
        "actions": actions,            # (num_seeds, num_algos, max_time_steps, num_agents, num_agents)
        "action_probs": action_probs,  # (num_seeds, num_algos, max_time_steps, num_agents, num_agents)
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

