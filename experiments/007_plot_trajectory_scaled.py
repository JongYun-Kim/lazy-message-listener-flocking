import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import pickle
import os


def get_save_path_and_file_name(date_str: str, start_seed: int, last_seed: int):
    # Get path
    base_path = "./../data/"
    dir_path = os.path.join(base_path, date_str + "/")
    #
    seed_range_str = f"seed_{start_seed}-{last_seed}_"
    plk_name = "002_acs_rl_nature_" + seed_range_str + date_str + ".pkl"

    return dir_path, plk_name


if __name__ == "__main__":
    # Get path
    folder_path, file_name = get_save_path_and_file_name("2024-03-16_16-08-55", 120, 619)

    # Load and check
    with open(os.path.join(folder_path, file_name), "rb") as f:
        data_loaded = pickle.load(f)
    print("Data loaded and checked")
    print(f"data keys: {data_loaded.keys()}")
    print(f"trajectories:     {data_loaded['trajectories'].shape}")
    print(f"algo_str: {data_loaded['algo_str']}")
    print(f"seeds:    {data_loaded['seeds']}")

    # Assign explicit variables
    trajectories = data_loaded['trajectories']  # (num_seeds, num_algos, max_time_steps, num_agents, 2)
    algo_str = data_loaded['algo_str']  # (num_algos)
    seeds = data_loaded['seeds']  # (num_seeds)

    # Dimensions
    num_seeds = len(seeds)
    num_algos = len(algo_str)
    max_time_steps = trajectories.shape[2]
    num_agents = trajectories.shape[3]

    # Plot settings
    my_dpi = 300

    # Get time
    times = np.arange(max_time_steps)

    # For each seed
    for episode in range(num_seeds):
        if seeds[episode] != 154:  # 특정 번호만 그리기
            continue

        if episode % 100 == 0:
            print(f"Episode {episode + 1}/{num_seeds}")

        # Find algorithms to plot (exclude topology and metric)
        plot_algos = []
        for algo_index in range(num_algos):
            if not (algo_str[algo_index].startswith("topology") or algo_str[algo_index].startswith("metric")):
                plot_algos.append(algo_index)

        if len(plot_algos) == 0:
            print("No algorithms to plot")
            continue

        # Calculate global min/max for consistent scaling across subplots
        all_x_coords = []
        all_y_coords = []

        for algo_index in plot_algos:
            for agent_idx in range(num_agents):
                agent_trajectory = trajectories[episode, algo_index, :, agent_idx, :2]
                all_x_coords.extend(agent_trajectory[:, 0])
                all_y_coords.extend(agent_trajectory[:, 1])

        # Calculate bounds with some padding
        x_min, x_max = min(all_x_coords), max(all_x_coords)
        y_min, y_max = min(all_y_coords), max(all_y_coords)

        # Add padding (2% of the range)
        x_padding = (x_max - x_min) * 0.02
        y_padding = (y_max - y_min) * 0.02

        x_min -= x_padding
        x_max += x_padding
        y_min -= y_padding
        y_max += y_padding

        # Create subplot layout based on number of algorithms
        n_plots = len(plot_algos)
        fig, axes = plt.subplots(1, n_plots, figsize=(5 * n_plots, 5))

        # Handle case when there's only one subplot
        if n_plots == 1:
            axes = [axes]

        # Plot trajectories
        for i, algo_index in enumerate(plot_algos):
            # Plot trajectory with gradient for each agent
            for agent_idx in range(num_agents):
                agent_trajectory = trajectories[episode, algo_index, :, agent_idx, :2]
                points = agent_trajectory.reshape(-1, 1, 2)
                segments = np.concatenate([points[:-1], points[1:]], axis=1)
                norm = plt.Normalize(times.min(), times.max())
                lc = LineCollection(segments, cmap='viridis', norm=norm)
                lc.set_array(times)
                lc.set_linewidth(2)
                line = axes[i].add_collection(lc)

            # Set consistent axis limits for all subplots
            axes[i].set_xlim(x_min, x_max)
            axes[i].set_ylim(y_min, y_max)
            axes[i].set_aspect('equal', adjustable='box')  # Force equal aspect ratio

            axes[i].set_title(algo_str[algo_index], fontsize=20)
            axes[i].tick_params(axis='both', which='major', labelsize=16)
            axes[i].grid(True, alpha=0.3)

        # First apply tight_layout, then adjust for colorbar
        plt.tight_layout()

        # Make space for colorbar at the bottom
        plt.subplots_adjust(bottom=0.2, wspace=0.1)

        # Add colorbar positioned at the bottom center
        cb = fig.colorbar(line, ax=axes, orientation='horizontal',
                          shrink=0.6, pad=0.15, aspect=25)
        cb.set_label('Time steps', fontsize=16)
        cb.ax.tick_params(labelsize=16)

        # Save the plot
        if not os.path.exists(folder_path + "plots/"):
            os.makedirs(folder_path + "plots/")
        plt.savefig(folder_path + "plots/" + f"trajectories_seed_{seeds[episode]}.png", dpi=my_dpi, bbox_inches='tight')
        plt.close()
        print(f"Trajectory plot {episode + 1}/{num_seeds} saved")

    print("done")