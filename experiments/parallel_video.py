import numpy as np
from env.envs import LazyMsgListenersEnv, ProposedFucking
import time
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation
# import os
import datetime as dt
import ray
import itertools


def get_default_config():
    num_agents_default = 80
    config_to_set_to_default = {
        "num_agents_pool": np.array([num_agents_default], dtype=np.int32),
        # "std_p_goal": 45,
        "std_v_goal": 0.15,
        "std_p_rate_goal": 0.15,
        "std_v_rate_goal": 0.3,
        "max_time_steps": 1500,
        #
        "comm_range": 2000,
        #
        "control_config": {
            "speed": 15,  # Speed in m/s. Default is 15
            "predefined_distance": 60,  # Predefined distance in meters. Default is 60
            "communication_decay_rate": 1/3,  # Communication decay rate. Default is 1/3
            "cost_weight": 1,  # Cost weight. Default is 1
            "inter_agent_strength": 5,  # Inter agent strength. Default is 5
            "bonding_strength": 1,  # Bonding strength. Default is 1
            "k1": 1,  # K1 coefficient. Default is 1
            "k2": 3,  # K2 coefficient. Default is 3
            "max_turn_rate": 8/15,  # Maximum turn rate in rad/s. Default is 8/15
            "initial_position_bound": 250,  # Initial position bound in meters. Default is 250
        },
        #
        "get_state_hist": True,
    }
    return config_to_set_to_default


def compute_actions(test_env):
    test_actions = np.ones((test_env.num_agents_max, test_env.num_agents_max), dtype=np.bool_)
    np.fill_diagonal(test_actions, 0)  # no self-communication
    test_actions = test_actions * test_env.state["neighbor_masks"]
    return test_actions


@ray.remote
def run_episode(episode_num, config, num_episodes):
    show_comm_range = False

    # env = LazyMsgListenersEnv(config)
    env = ProposedFucking(config)  ####################################################################
    loop_start_time = time.time()
    done = False
    env.reset()
    R = env.comm_range if show_comm_range else 0
    print(f"Episode: {episode_num+1} / {num_episodes}")
    episodic_reward = 0
    while not done:
        actions = compute_actions(env)
        obs, reward, done, info = env.step(actions)
        episodic_reward += reward

    print("    Episode Length:     ", env.time_step)
    print("    Episodic Reward:    ", episodic_reward)
    print("    Communication Lost: ", env.has_lost_comm)

    # Plotting and Video Saving Logic
    dpi = 100
    interval = 400
    fps = 20

    fig, ax = plt.subplots(figsize=(10, 10), dpi=dpi)  # Adjust fig-size and dpi as needed
    agent_positions = env.agent_states_hist[:env.time_step, :, :2]
    agent_velocities = env.agent_states_hist[:env.time_step, :, 2:4]

    ax.grid(True)
    ax.set_aspect('equal')
    ax.set_xlim(np.min(agent_positions[:, :, 0]), np.max(agent_positions[:, :, 0]) + 0.66 * R)
    ax.set_ylim(np.min(agent_positions[:, :, 1]), np.max(agent_positions[:, :, 1]) + 0.66 * R)
    # ax.set_title(f"CommSecured <{~env.has_lost_comm}>, (Ep {episode_num + 1}) - {env.num_agents_max} Agents")
    ax.set_title(f"{env.num_agents_max} UAVs, cR={R}, fR={env.control_config['predefined_distance']}")

    # Initialize scatter and quiver plots for each environment
    scatter = ax.scatter(agent_positions[0, :, 0], agent_positions[0, :, 1])
    quiver = ax.quiver(agent_positions[0, :, 0], agent_positions[0, :, 1],
                       agent_velocities[0, :, 0], agent_velocities[0, :, 1])

    # Initialize trajectories for each agent in each environment
    trajectories = []
    for _ in range(agent_positions.shape[1]):  # Loop over agents
        trajectories.append(ax.plot([], [], color='gray', linewidth=0.5)[0])

    # Initialize communication range circles
    if show_comm_range:  #############################################################
        comm_circles = []
        alpha_val = 0.13 if show_comm_range else 0
        for _ in range(env.num_agents_max):
            circle = plt.Circle((0, 0), env.comm_range, color='blue', alpha=alpha_val,
                                zorder=2)  # Default color and transparency
            ax.add_patch(circle)
            comm_circles.append(circle)
        # alpha_val = 0.45 if show_comm_range else 0
        # for _ in range(env.num_agents_max):
        #     circle = patches.Circle((0, 0), env.comm_range, color='blue', alpha=alpha_val, fill=False, zorder=2, linewidth=1.5)
        #     ax.add_patch(circle)
        #     comm_circles.append(circle)

    # Update function
    def update(frame):
        components = []
        if frame < env.time_step:  # Check if the environment is still active
            agent_positions = env.agent_states_hist[frame, :, 0:2]
            agent_velocities = env.agent_states_hist[frame, :, 2:4]
        else:  # If done, use the last available frame
            agent_positions = env.agent_states_hist[env.time_step - 1, :, 0:2]
            agent_velocities = env.agent_states_hist[env.time_step - 1, :, 2:4]

        # Update scatter (positions) with the highest zorder
        scatter.set_offsets(agent_positions)
        scatter.set_zorder(3)  # allows to be drawn in that order (smaller: bottom/first, larger: top/last)
        components.append(scatter)

        # Update velocities with a middle zorder
        quiver.set_offsets(agent_positions)
        quiver.set_UVC(agent_velocities[:, 0], agent_velocities[:, 1])
        quiver.set_zorder(4)
        components.append(quiver)

        # Update trajectories with a lower zorder
        for j, line in enumerate(trajectories):
            if frame < env.time_step:
                traj_data = env.agent_states_hist[:frame + 1, j, 0:2]
            else:
                traj_data = env.agent_states_hist[:env.time_step, j, 0:2]
            line.set_data(traj_data[:, 0], traj_data[:, 1])
            line.set_zorder(1)
            components.append(line)

        if show_comm_range:  #############################################################
            # Update communication range circles
            for k, circle in enumerate(comm_circles):
                if frame < env.time_step:
                    position = env.agent_states_hist[frame, k, 0:2]
                    comm_status = info["comm_loss_agents"][k]
                    circle.set_center(position)
                    circle.set_color('red' if comm_status else 'green')  # Red for lost comm, green otherwise
                else:
                    position = env.agent_states_hist[env.time_step - 1, k, 0:2]
                    circle.set_center(position)
                components.append(circle)

        return components

    # Animate
    max_time_step_video = env.time_step
    ani = FuncAnimation(fig, update, frames=max_time_step_video, interval=interval, blit=True)
    current_time = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    ani.save(f'./videos/{env.num_agents_max}UAVs_E_{episode_num + 1}_T_{current_time}.mp4', dpi=dpi, fps=fps, writer='ffmpeg')

    loop_duration = time.time() - loop_start_time
    print(f"    EnvDuration: {loop_duration} seconds")

    return episodic_reward, env.time_step, loop_duration, env.has_lost_comm


def generate_config_combinations(default_config, test_config):
    """
    Generate a list of config dictionaries from a test_config_dict and a default_config.
    :param test_config: A dictionary of config parameters to be tested. If the value is a list, then the parameter
    will be tested for each value in the list. If the value is a single value, then the parameter will be set to that
    value for all the generated config dictionaries.
    :param default_config: A dictionary of default config parameters. This dictionary will be used to generate the
    config dictionaries.
    :return: A list of config dictionaries of all the combinations of the test cases.
    """
    def generate_combinations(current_path, config):
        if isinstance(config, dict):
            for key, value in config.items():
                yield from generate_combinations(current_path + [key], value)
        elif isinstance(config, list):
            yield current_path, config

    def set_value_in_nested_dict(dictionary, keys, value):
        for key in keys[:-1]:
            dictionary = dictionary.setdefault(key, {})
        dictionary[keys[-1]] = value

    combinations = list(generate_combinations([], test_config))
    keys_list = [item[0] for item in combinations]
    values_list = [item[1] for item in combinations]

    all_combinations = itertools.product(*values_list)

    config_list = []
    for combination in all_combinations:
        new_config = dict(default_config)
        # Deep copy nested dictionaries to avoid shared reference issues
        for key in new_config.keys():
            if isinstance(new_config[key], dict):
                new_config[key] = dict(new_config[key])

        for keys, value in zip(keys_list, combination):
            set_value_in_nested_dict(new_config, keys, value)
        config_list.append(new_config)

    return config_list


def convert_list_to_numpy_each(num_agents_pool_in_list):
    num_agents_pool_in_numpy = []
    for i, num_agents in enumerate(num_agents_pool_in_list):
        num_agents_pool_in_numpy.append(np.array([num_agents], dtype=np.int32))
    return num_agents_pool_in_numpy


if __name__ == "__main__":
    default_config = get_default_config()
    test_config_dict = {
        "num_agents_pool": convert_list_to_numpy_each([10, 15, 20, 25, 30, 35, 40, 45, 50, 55]),
        # "comm_range": [80, 100, 120, 150, 200],
        "control_config": {
            # "speed": [15, 20],
            # "predefined_distance": [60, 120, 200],
            # "max_turn_rate": [0.8, 0.6, 0.4],
            # "initial_position_bound": [250, 500, 800]
        }
    }
    test_config_dict_list = generate_config_combinations(default_config, test_config_dict)
    num_test_cases = len(test_config_dict_list)
    num_repeats = 3
    num_episodes = num_test_cases * num_repeats

    # Initialize Ray
    ray.init(num_cpus=15)
    # ray.init(local_mode=True)

    # Parallel execution
    ray_start_time = time.time()
    futures = [run_episode.remote(i, test_config_dict_list[i // num_repeats], num_episodes) for i in range(num_episodes)]

    results = ray.get(futures)
    ray_duration = time.time() - ray_start_time

    # Gather results
    episodic_rewards = np.zeros(num_episodes)
    episode_lengths = np.zeros(num_episodes)
    loop_durations = np.zeros(num_episodes)
    has_lost_comm = np.zeros(num_episodes)
    for i, result in enumerate(results):
        episodic_rewards[i] = result[0]
        episode_lengths[i] = result[1]
        loop_durations[i] = result[2]
        has_lost_comm[i] = result[3]
    num_failure = np.sum(has_lost_comm)
    success_rate = 1 - num_failure / num_episodes
    success_rewards = episodic_rewards[has_lost_comm == 0]
    success_episode_lengths = episode_lengths[has_lost_comm == 0]
    success_loop_durations = loop_durations[has_lost_comm == 0]

    print("All episodes completed.")
    print(f"Ray Duration: {ray_duration} seconds")
    print("----------------------------------------")
    print(f"------Results for {num_episodes} episodes------")
    print(f"Average All Episodic Reward: {np.mean(episodic_rewards)}")
    print(f"Average All Episode Length:  {np.mean(episode_lengths)}")
    print(f"Average All Loop Duration:   {np.mean(loop_durations)}")
    print("----------------------------------------")
    print(f"comm_range: {default_config['comm_range']}")
    print(f"Number of Failure: {num_failure} out of {num_episodes} episodes")
    print(f"Success Rate: {success_rate}")
    print(f"Average Success Episodic Reward: {np.mean(success_rewards)}")
    print(f"Average Success Episode Length:  {np.mean(success_episode_lengths)}")
    print(f"Average Success Loop Duration:   {np.mean(success_loop_durations)}")
    print("----------------------------------------")

    print("Done")
