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
    pkl_name = "001_acs_vs_rl_" + seed_range_str + date_str + ".pkl"

    return dir_path, pkl_name


if __name__ == "__main__":
    """
    # Data settings
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
    """
    # Get path
    # folder_path, file_name = get_save_path_and_file_name("2024-03-12_14-13-31", 0, 9)
    # folder_path, file_name = get_save_path_and_file_name("2024-03-12_16-35-04", 10, 19)
    folder_path, file_name = get_save_path_and_file_name("2024-03-13_13-48-07", 20, 119)

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
    # # And save the plots in folder_path + "plots/"
    my_dpi = 300  # no arg: 100 (None), 300 is good for printing
    algo_colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]  # blue, orange, green
    assert len(algo_colors) == num_algos

    # Get time
    times = np.arange(max_time_steps)

    # For each seed
    for episode in range(num_seeds):
        # Plot 1: Trajectories in subplots (1, num_algos);  use cmap='viridis' to gradient spectrum on the temporal direction on the trajectory
        # Create a new figure with subplots
        fig, axes = plt.subplots(1, num_algos, figsize=(15, 5), constrained_layout=True)
        for algo_index in range(num_algos):
            # Plot trajectory with gradient
            for agent_idx in range(num_agents):
                agent_trajectory = trajectories[episode, algo_index, :, agent_idx, :2]
                points = agent_trajectory.reshape(-1, 1, 2)
                segments = np.concatenate([points[:-1], points[1:]], axis=1)
                norm = plt.Normalize(times.min(), times.max())
                lc = LineCollection(segments, cmap='viridis', norm=norm)
                lc.set_array(times)
                lc.set_linewidth(2)
                line = axes[algo_index].add_collection(lc)
                axes[algo_index].set_title(algo_str[algo_index])
                axes[algo_index].axis('equal')
        # Add colorbar
        # cb = fig.colorbar(line, orientation='horizontal')
        cb = fig.colorbar(line, orientation='vertical')
        cb.set_label('Time steps')
        # Adjust the layout so that the plots do not overlap
        # plt.tight_layout()
        # Save the plot
        # Check if the folder exists
        if not os.path.exists(folder_path + "plots/"):
            os.makedirs(folder_path + "plots/")
        plt.savefig(folder_path + "plots/" + f"trajectories_seed_{seeds[episode]}.png", dpi=my_dpi)
        plt.close()
        print(f"Trajectory plot {episode+1}/{num_seeds} saved")

        #
        # Plot 2: Spatial and velocity entropy in subplots (2, 1); each subplot includes all algorithms as lines
        # Create a new figure with subplots
        fig, axes = plt.subplots(2, 1, figsize=(10, 10), constrained_layout=True)
        for algo_index in range(num_algos):
            # Plot with line width set to 2.0
            axes[0].plot(times, spatial_entropy[episode, algo_index, :], label=algo_str[algo_index],
                         color=algo_colors[algo_index], linewidth=2.0)
            axes[1].plot(times, velocity_entropy[episode, algo_index, :], label=algo_str[algo_index],
                         color=algo_colors[algo_index], linewidth=2.0)

        # Set labels and titles
        axes[0].set_xlabel('Time steps')
        axes[0].set_ylabel('Spatial Entropy')
        axes[0].set_title('Spatial Entropy Over Time')
        axes[0].legend()
        axes[1].set_xlabel('Time')
        axes[1].set_ylabel('Velocity Entropy')
        axes[1].set_title('Velocity Entropy Over Time')
        axes[1].legend()

        # Set grid with subtle color (with transparency)
        axes[0].grid(color='gray', linestyle='--', linewidth=0.5, alpha=0.67)
        axes[1].grid(color='gray', linestyle='--', linewidth=0.5, alpha=0.67)

        # Ensure x axes start with 0
        axes[0].set_xlim(left=0)
        axes[1].set_xlim(left=0)
        # Start the y-axis with 0 only for the velocity entropy plot
        axes[1].set_ylim(bottom=0)

        # Save and close the plot
        plt.savefig(folder_path + "plots/" + f"entropy_seed_{seeds[episode]}.png", dpi=my_dpi)
        plt.close()
        print(f"Entropy plot {episode+1}/{num_seeds} saved")

        # Plot 3: 'Control/u_sat' in one plot; each algorithm has its own line (reward includes the control cost)
        # Create a new figure with subplots
        u_sat = 8/15  # 8/15 rad/s
        e_to_u = 3/2
        fig, axes = plt.subplots(1, 1, figsize=(10, 5), constrained_layout=True)
        for algo_index in range(num_algos):
            # Plot with line width set to 2.0
            axes.plot(times, -control[episode, algo_index, :]/(u_sat*e_to_u), label=algo_str[algo_index],
                      color=algo_colors[algo_index], linewidth=2.0)
        # Set labels and titles
        axes.set_xlabel('Time steps')
        axes.set_ylabel('Normalized average control input')
        axes.set_title('Control Input Over Time')
        axes.legend()
        # Set grid with subtle color (with transparency)
        axes.grid(color='gray', linestyle='--', linewidth=0.5, alpha=0.67)
        # Ensure x axes start with 0
        axes.set_xlim(left=0)
        # Ensure y axes start with 0 and end with 1
        axes.set_ylim(bottom=0, top=1)
        # Save and close the plot
        plt.savefig(folder_path + "plots/" + f"control_seed_{seeds[episode]}.png", dpi=my_dpi)
        plt.close()
        print(f"Control plot {episode+1}/{num_seeds} saved")

        # Plot 4: Average Action in one plot; each algorithm has its own line; np.sum(actions[episode, algo_index, t, :, :]) / (num_agents)**2
        # Create a new figure with subplots
        fig, axes = plt.subplots(1, 1, figsize=(10, 5), constrained_layout=True)
        for algo_index in range(num_algos):
            # Plot with line width set to 2.0
            average_action = np.zeros(max_time_steps)
            for t in range(max_time_steps):
                average_action[t] = (np.sum(actions[episode, algo_index, t, :, :])-num_agents) / ((num_agents-1)*num_agents)
            axes.plot(times, average_action, label=algo_str[algo_index],
                      color=algo_colors[algo_index], linewidth=2.0)
        # Set labels and titles
        axes.set_xlabel('Time steps')
        axes.set_ylabel('Average neighbor selection probability')
        axes.set_title('Average Neighbor Selection Probability Over Time')
        axes.legend()
        # Set grid with subtle color (with transparency)
        axes.grid(color='gray', linestyle='--', linewidth=0.5, alpha=0.67)
        # Ensure x axes start with 0
        axes.set_xlim(left=0)
        # Ensure y axes start with 0 and end with 1
        axes.set_ylim(bottom=0, top=1.12)
        # Save and close the plot
        plt.savefig(folder_path + "plots/" + f"average_action_seed_{seeds[episode]}.png", dpi=my_dpi)
        plt.close()
        print(f"Average action plot {episode+1}/{num_seeds} saved")


    print("done")

