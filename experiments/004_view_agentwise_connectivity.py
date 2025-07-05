import numpy as np
import pickle
import os
from datetime import datetime
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt


def transform_data(data, ego_index):
    transformed_data = np.zeros_like(data)
    for t in range(data.shape[0]):
        ego_pos = data[t, ego_index, :2]
        ego_heading = data[t, ego_index, 2]
        translated_positions = data[t, :, :2] - ego_pos
        rotation_matrix = np.array([[np.cos(-ego_heading), -np.sin(-ego_heading)],
                                    [np.sin(-ego_heading), np.cos(-ego_heading)]])
        for i in range(data.shape[1]):
            transformed_data[t, i, :2] = np.dot(rotation_matrix, translated_positions[i])
            transformed_data[t, i, 2] = data[t, i, 2] - ego_heading
    return transformed_data


def calculate_fixed_limit(transformed_data):
    max_abs_coord = np.max(np.abs(transformed_data[:, :, :2]))
    padding = max_abs_coord * 0.1
    limit = max_abs_coord + padding
    return limit


def animate(i, ego_index, scatters, ego_scatters, quivers, lines, transformed_data, connectivity, connectivity_probs, limit, axs):
    # Loop through each subplot to update scatters, quivers, and lines
    for ax_index, ax in enumerate(axs):
        # Update positions for all agents and the ego agent
        scatters[ax_index].set_offsets(transformed_data[i, :, :2])
        ego_scatters[ax_index].set_offsets(transformed_data[i, ego_index:ego_index + 1, :2])
        # Update quiver for all agents
        quivers[ax_index].set_offsets(transformed_data[i, :, :2])
        quivers[ax_index].set_UVC(np.cos(transformed_data[i, :, 2]), np.sin(transformed_data[i, :, 2]))

        # Remove previous lines
        for line in lines[ax_index]:
            line.remove()
        lines[ax_index].clear()

        # Draw new lines for connections
        for agent_index in range(transformed_data.shape[1]):
            if agent_index != ego_index:
                if ax_index == 0:  # ax[0] uses connectivity_probs for alpha values
                    alpha = connectivity_probs[i, agent_index]
                else:  # ax[1] uses connectivity for alpha values
                    alpha = connectivity[i, agent_index]

                alpha = 0  # TODO: Remove this line to use the actual alpha values
                line = ax.plot([transformed_data[i, ego_index, 0], transformed_data[i, agent_index, 0]],
                               [transformed_data[i, ego_index, 1], transformed_data[i, agent_index, 1]],
                               color="green", alpha=alpha)
                lines[ax_index].append(line[0])

        # Update scatter plot colors based on connectivity or connectivity_probs
        if ax_index == 0:
            scatters[ax_index].set_array(connectivity_probs[i])
        else:
            scatters[ax_index].set_array(connectivity[i])

        ax.set_xlim([-limit, limit])
        ax.set_ylim([-limit, limit])


def create_animation_for_ego(data, ego_index, connectivity, connectivity_probs, seed):
    # data: (max_time_steps, num_agents, 3)  # positions and headings (x, y, theta)
    # connectivity: (max_time_steps, num_agents)
    # connectivity_probs: (max_time_steps, num_agents)

    transformed_data = transform_data(data, ego_index)
    fixed_limit = calculate_fixed_limit(transformed_data)
    fig, axs = plt.subplots(1, 2, figsize=(16, 8))

    scatters = []
    ego_scatters = []
    quivers = []
    lines = [[], []]  # To store lines for each subplot

    # Initialize scatter, ego_scatter, and quiver for each subplot
    for ax_index, ax in enumerate(axs):
        ax.set_facecolor('darkgrey')  # Set the background color to light grey
        ax.set_aspect('equal', 'box')
        scatter = ax.scatter(transformed_data[0, :, 0], transformed_data[0, :, 1], s=90,
                             c=connectivity_probs[0] if ax_index == 0 else connectivity[0], cmap='hot', vmin=0, vmax=1)
        ego_scatter = ax.scatter(transformed_data[0, ego_index, 0], transformed_data[0, ego_index, 1], s=90, color='slateblue')
        quiver = ax.quiver(transformed_data[0, :, 0], transformed_data[0, :, 1], np.cos(transformed_data[0, :, 2]), np.sin(transformed_data[0, :, 2]),
                           angles='xy', scale_units='xy', scale=0.1, color='k')
        scatters.append(scatter)
        ego_scatters.append(ego_scatter)
        quivers.append(quiver)

    # Add a colorbar as a heatmap bar
    for ax in axs:
        cbar = plt.colorbar(scatters[0], ax=ax)
        cbar.set_label('Connectivity')

    # anim = FuncAnimation(fig, animate,  frames=200,  # frames=data.shape[0],
    anim=FuncAnimation(fig, animate, frames=data.shape[0],
                         fargs=(ego_index, scatters, ego_scatters, quivers, lines, transformed_data, connectivity,
                                connectivity_probs, fixed_limit, axs),
                         interval=100)
    file_name = f"seed_{seed}_ego_{ego_index}_connectivity.mp4"
    anim.save(file_name, writer='ffmpeg', fps=15, dpi=300, # Reduced fps to slow down the animation
              progress_callback=lambda i, n: print(f'Progress {i / n * 100:.2f}%'))
    plt.close(fig)


if  __name__ == "__main__":
    # Get file path
    file_path = "./../data/2024-03-25_17-00-08/plot_004_acs_rl_seed_120-135.pkl"
    # Load the pickle file
    with open(file_path, "rb") as f:
        data_loaded = pickle.load(f)
    print("Pickle file has been successfully loaded:")
    print(f"    from {file_path}")

    # Extract the data
    trajectories = data_loaded['trajectories']  # (num_seeds, num_algos, max_time_steps, num_agents, 2)
    velocities = data_loaded['velocities']      # (num_seeds, num_algos, max_time_steps, num_agents, 2)
    spatial_entropy = data_loaded['spatial_entropy']    # (num_seeds, num_algos, max_time_steps)
    velocity_entropy = data_loaded['velocity_entropy']  # (num_seeds, num_algos, max_time_steps)
    actions = data_loaded['actions']  # (num_seeds, num_algos, max_time_steps, num_agents, num_agents)
    action_probs = data_loaded['action_probs']  # (num_seeds, num_algos, max_time_steps, num_agents, num_agents)
    algo_str = data_loaded['algo_str']  # (num_algos)
    seeds = data_loaded['seeds']        # (num_seeds)

    num_seeds, num_algos, max_time_steps, num_agents, _ = trajectories.shape

    for episode in range(num_seeds):
        algo_idx = 1  # RL

        trajs = trajectories[episode, algo_idx, :, :, :]  # (max_time_steps, num_agents, 2)
        vels = velocities[episode, algo_idx, :, :, :]  # (max_time_steps, num_agents, 2)
        spa_ents = spatial_entropy[episode, algo_idx, :]  # (max_time_steps)
        vel_ents = velocity_entropy[episode, algo_idx, :]  # (max_time_steps)
        act_sampled = actions[episode, algo_idx, :, :, :]  # (max_time_steps, num_agents, num_agents)
        act_probs = action_probs[episode, algo_idx, :, :, :]  # (max_time_steps, num_agents, num_agents)

        positions = trajs
        headings = np.arctan2(vels[:, :, 1], vels[:, :, 0])  # (max_time_steps, num_agents)
        # trajectory consists of positions and headings of agents
        trajectory = np.concatenate([positions, headings[:, :, None]], axis=-1)  # (max_time_steps, num_agents, 3)





        # Use the function for the desired ego_index, assuming trajectory is defined
        if episode != 1:
            continue
        agent_idx = 4
        create_animation_for_ego(trajectory, agent_idx, act_sampled[:,agent_idx,:], act_probs[:,agent_idx,:], seeds[episode])
        # Loop over each ego to create and save an animation
        # for agent_idx in range(num_agents):
        #     create_animation_for_ego(trajectory, agent_idx, act_sampled[:,agent_idx,:], act_probs[:,agent_idx,:], seeds[episode])

    