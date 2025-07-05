import numpy as np
# matplotlib
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib import cm
# Save files and load files
import pickle
import os  # creates dirs
from datetime import datetime  # gets current date and time
from ray.tune.logger import pretty_print


def get_save_path_and_file_name(date_str: str, start_seed: int, last_seed: int):
    # Get path
    base_path = "./../data/"
    dir_path = os.path.join(base_path, date_str + "/")
    #
    seed_range_str = f"seed_{start_seed}-{last_seed}_"
    plk_name = "005_random_" + seed_range_str + date_str + ".pkl"

    return dir_path, plk_name


if __name__ == "__main__":
    """
    # Data settings
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
        "originated_from": "002_collect_compare_RL-ACS-metric-topology-random.py",
    }
    """
    # Get path
    # folder_path, file_name = get_save_path_and_file_name("2024-05-08_01-03-10", 120, 129)
    # folder_path, file_name = get_save_path_and_file_name("2024-05-08_01-38-11", 120, 619)
    folder_path, file_name = get_save_path_and_file_name("2024-05-08_09-27-47", 120, 619)

    # Load and check
    with open(os.path.join(folder_path, file_name), "rb") as f:
        data_loaded = pickle.load(f)
    print("Data loaded and checked")
    print(f"data keys: {data_loaded.keys()}")
    print(f"trajectories:     {data_loaded['trajectories'].shape}")
    print(f"velocities:       {data_loaded['velocities'].shape}")
    print(f"spatial_entropy:  {data_loaded['spatial_entropy'].shape}")
    print(f"velocity_entropy: {data_loaded['velocity_entropy'].shape}")
    print(f"actions:          {data_loaded['actions'].shape}")
    print(f"action_probs:     {data_loaded['action_probs'].shape}")
    print(f"rewards:          {data_loaded['rewards'].shape}")
    print(f"control:          {data_loaded['control'].shape}")
    print(f"algo_str: {data_loaded['algo_str']}")
    print(f"seeds:    {data_loaded['seeds']}")
    print(f"originated_from: {data_loaded['originated_from']}")
    print("Data loaded successfully")

    # Assign explicit variables
    trajectories = data_loaded['trajectories']  # (num_seeds, num_algos, max_time_steps, num_agents, 2)
    velocities = data_loaded['velocities']      # (num_seeds, num_algos, max_time_steps, num_agents, 2)
    spatial_entropy = data_loaded['spatial_entropy']    # (num_seeds, num_algos, max_time_steps)
    velocity_entropy = data_loaded['velocity_entropy']  # (num_seeds, num_algos, max_time_steps)
    actions = data_loaded['actions']  # (num_seeds, num_algos, max_time_steps, num_agents, num_agents)
    action_probs = data_loaded['action_probs']  # (num_seeds, num_algos, max_time_steps, num_agents, num_agents)
    rewards = data_loaded['rewards']  # (num_seeds, num_algos, max_time_steps)
    control = data_loaded['control']  # (num_seeds, num_algos, max_time_steps)
    algo_str = data_loaded['algo_str']  # (num_algos)
    seeds = data_loaded['seeds']        # (num_seeds)

    # Dimensions
    num_seeds = len(seeds)
    num_algos = len(algo_str)
    max_time_steps = spatial_entropy.shape[2]
    num_agents = trajectories.shape[3]

    # Plot settings (Draw four different plots)
    # # Plot 1: Trajectories in subplots (1, num_algos);
    # # Plot 2: Spatial and velocity entropy in subplots (2, 1)
    # # Plot 3: Control/u_sat in one plot
    # # Plot 4: Average Action in one plot
    # # Plot 5: Average Action Prob in one plot
    # # And save the plots in folder_path + "plots/"
    my_dpi = 300  # no arg: 100 (None), 300 is good for printing

    # Get time
    times = np.arange(max_time_steps)

    # Save the plot
    plot_folder_path = os.path.join(folder_path, "plots")
    # Check if the folder exists
    if not os.path.exists(plot_folder_path):
        os.makedirs(plot_folder_path)

    # For each seed
    for episode in range(num_seeds):
        # if seeds[episode] != 154:
        #     continue

        if episode % 100 == 0:
            print(f"Episode {episode + 1}/{num_seeds}")
        # Plot 1: Trajectories in subplots (1, num_algos);  use cmap='viridis' to gradient spectrum on the temporal direction on the trajectory
        # # Create a new figure with subplots
        # # fig, axes = plt.subplots(1, 2, figsize=(10, 5), constrained_layout=True)
        # fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        # for algo_index in range(num_algos):
        #     if algo_str[algo_index].startswith("topology"):
        #         continue
        #     if algo_str[algo_index].startswith("metric"):
        #         continue
        #     # Plot trajectory with gradient
        #     for agent_idx in range(num_agents):
        #         agent_trajectory = trajectories[episode, algo_index, :, agent_idx, :2]
        #         points = agent_trajectory.reshape(-1, 1, 2)
        #         segments = np.concatenate([points[:-1], points[1:]], axis=1)
        #         norm = plt.Normalize(times.min(), times.max())
        #         lc = LineCollection(segments, cmap='viridis', norm=norm)
        #         lc.set_array(times)
        #         lc.set_linewidth(2)
        #         line = axes[algo_index].add_collection(lc)
        #         axes[algo_index].set_title(algo_str[algo_index], fontsize=20)  # Set larger font for title
        #         axes[algo_index].tick_params(axis='both', which='major', labelsize=16)  # Set larger font for ticks
        #         axes[algo_index].axis('equal')
        # # Add colorbar
        # # Define colorbar axes
        # # Add colorbar to the defined axes and set its orientation to horizontal
        # cb = fig.colorbar(line, orientation='horizontal', ax=axes)
        # cb.set_label('Time steps', fontsize=16)  # Set larger font for colorbar label
        # cb.ax.tick_params(labelsize=16)  # Set larger font for colorbar ticks
        # # plt.tight_layout()
        # # Save the plot
        # # # Check if the folder exists
        # # if not os.path.exists(folder_path + "plots/"):
        # #     os.makedirs(folder_path + "plots/")
        # plt.savefig(folder_path + "plots/" + f"trajectories_seed_{seeds[episode]}.png", dpi=my_dpi)
        # plt.close()
        # print(f"Trajectory plot {episode+1}/{num_seeds} saved")
        #
        for algo_index, algo_name in enumerate(algo_str):
            # Create a new figure for each algorithm
            fig, ax = plt.subplots(figsize=(5, 5), constrained_layout=True)
            for agent_idx in range(num_agents):
                agent_trajectory = trajectories[episode, algo_index, :, agent_idx, :2]
                points = agent_trajectory.reshape(-1, 1, 2)
                segments = np.concatenate([points[:-1], points[1:]], axis=1)
                norm = plt.Normalize(times.min(), times.max())
                lc = LineCollection(segments, cmap='viridis', norm=norm)
                lc.set_array(times)
                lc.set_linewidth(2)
                line = ax.add_collection(lc)
                ax.set_title(f"{algo_name} - {num_agents}agents - {seeds[episode]}")
                ax.axis('equal')
            # Add colorbar
            cb = fig.colorbar(line, orientation='vertical')
            cb.set_label('Time steps')

            # Save the plot
            plot_folder_path = os.path.join(folder_path, "plots")
        #     # Check if the folder exists
        #     if not os.path.exists(plot_folder_path):
        #         os.makedirs(plot_folder_path)
            plt.savefig(os.path.join(plot_folder_path,
                                     f"trajectory_{algo_name}_seed_{seeds[episode]}_{num_agents}agent.png"),
                        dpi=my_dpi)
            plt.close()
            print(
                f"  Trajectory plot for Seed {episode + 1}/{num_seeds}, algorithm {algo_index + 1}/{num_algos} saved")

        #
        # Plot 2: Spatial and velocity entropy in subplots (2, 1); each subplot includes all algorithms as lines
        # Create a new figure with subplots
        # fig, axes = plt.subplots(2, 1, figsize=(10, 10), constrained_layout=True)
        fig, axes = plt.subplots(2, 1, figsize=(12, 10))
        for algo_index in range(num_algos):
            # Plot with line width set to 2.0
            axes[0].plot(times, spatial_entropy[episode, algo_index, :], label=algo_str[algo_index], linewidth=2.0)
            axes[1].plot(times, velocity_entropy[episode, algo_index, :], label=algo_str[algo_index], linewidth=2.0)

        # Set labels and titles
        axes[0].set_xlabel('Time steps', fontsize=18)
        axes[0].set_ylabel('Spatial Entropy', fontsize=18)
        axes[0].set_title('Spatial Entropy Over Time', fontsize=20)
        axes[0].tick_params(axis='both', which='major', labelsize=18)  # Set larger font for ticks
        # Let's place the legend outside the plot
        # axes[0].legend(loc='upper right', bbox_to_anchor=(0.97, 0.97), fontsize=24)
        # axes[1].legend(loc='upper right', bbox_to_anchor=(0.97, 0.97), fontsize=24)
        axes[1].set_xlabel('Time steps', fontsize=18)
        axes[1].set_ylabel('Velocity Entropy', fontsize=18)
        axes[1].set_title('Velocity Entropy Over Time', fontsize=20)
        axes[1].tick_params(axis='both', which='major', labelsize=18)
        axes[1].legend(loc='upper center', bbox_to_anchor=(0.5, -0.17), ncol=4, fontsize=18)

        # Set grid with subtle color (with transparency)
        axes[0].grid(color='gray', linestyle='--', linewidth=0.5, alpha=0.72)
        axes[1].grid(color='gray', linestyle='--', linewidth=0.5, alpha=0.72)

        # Ensure x axes start with 0
        axes[0].set_xlim(left=0, right=max_time_steps)
        axes[1].set_xlim(left=0, right=max_time_steps)
        # Limit the y-axis range from 30 to 110 for the spatial entropy plot
        axes[0].set_ylim(bottom=35, top=120)
        # Start the y-axis with 0 only for the velocity entropy plot
        axes[1].set_ylim(bottom=0)

        plt.tight_layout()

        # Save and close the plot
        plt.savefig(folder_path + "plots/" + f"entropy_random_seed_{seeds[episode]}.png", dpi=my_dpi)
        plt.close()
        print(f"Entropy plot {episode + 1}/{num_seeds} saved")

    print("Pause here to check the data")

