import numpy as np
from env.envs import LazyMsgListenersEnv
import time
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation
# import os
import datetime as dt


def get_config():
    num_agents = 20
    config_to_set = {
        "num_agents_pool": [num_agents],
        "std_p_goal": 50,
        "std_v_goal": 0.15,
        "std_p_rate_goal": 0.15,
        "std_v_rate_goal": 0.3,
        "max_time_steps": 1000,
        #
        "comm_range": 80,
        #
        "control_config": {
            "speed": 12,
            "predefined_distance": 50,
            "max_turn_rate": 0.8,  #8/15,
            "initial_position_bound": 200,
        },
        #
        "get_state_hist": True,
    }
    return config_to_set


def compute_actions(test_env):
    test_actions = np.ones((test_env.num_agents_max, test_env.num_agents_max), dtype=np.bool_)
    np.fill_diagonal(test_actions, 0)  # no self-communication
    test_actions = test_actions * env.state["neighbor_masks"]
    return test_actions


if __name__ == "__main__":
    num_episodes = 2
    # fig settings
    dpi = 100
    interval = 400
    fps = 20

    config = get_config()
    env = LazyMsgListenersEnv(config)
    # actions = compute_actions(env)
    episodic_reward_sum = 0
    for i in range(num_episodes):
        loop_start_time = time.time()
        done = False
        env.reset()
        print(f"Episode: {i+1} / {num_episodes}")
        reward_sum = 0
        while not done:
            actions = compute_actions(env)
            obs, reward, done, info = env.step(actions)
            reward_sum += reward
        print("    Episode Length:     ", env.time_step)
        print("    Episodic Reward:    ", reward_sum)
        print("    Communication Lost: ", env.has_lost_comm)
        loop_duration = time.time() - loop_start_time
        episodic_reward_sum += reward_sum
        print(f"    EnvDuration: {loop_duration} seconds")

        fig, ax = plt.subplots(figsize=(10, 10), dpi=dpi)  # Adjust fig-size and dpi as needed
        agent_positions = env.agent_states_hist[:env.time_step, :, :2]
        agent_velocities = env.agent_states_hist[:env.time_step, :, 2:4]

        ax.grid(True)
        ax.set_aspect('equal')
        ax.set_xlim(np.min(agent_positions[:, :, 0]), np.max(agent_positions[:, :, 0]))
        ax.set_ylim(np.min(agent_positions[:, :, 1]), np.max(agent_positions[:, :, 1]))
        ax.set_title(f"CommSecured <{~env.has_lost_comm}>, (Ep {i+1}) - {env.num_agents_max} Agents")

        # Initialize scatter and quiver plots for each environment
        scatter = ax.scatter(agent_positions[0, :, 0], agent_positions[0, :, 1])
        quiver = ax.quiver(agent_positions[0, :, 0], agent_positions[0, :, 1],
                           agent_velocities[0, :, 0], agent_velocities[0, :, 1])

        # Initialize trajectories for each agent in each environment
        trajectories = []
        for _ in range(agent_positions.shape[1]):  # Loop over agents
            trajectories.append(ax.plot([], [], color='gray', linewidth=0.5)[0])

        # Initialize communication range circles
        comm_circles = []
        # for _ in range(env.num_agents_max):
        #     circle = plt.Circle((0, 0), env.comm_range, color='blue', alpha=0.1,
        #                         zorder=2)  # Default color and transparency
        #     ax.add_patch(circle)
        #     comm_circles.append(circle)
        for _ in range(env.num_agents_max):
            circle = patches.Circle((0, 0), env.comm_range, color='blue', alpha=0.4, fill=False, zorder=2, linewidth=1.3)
            ax.add_patch(circle)
            comm_circles.append(circle)

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
        current_time = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
        max_time_step_video = env.time_step
        ani = FuncAnimation(fig, update, frames=max_time_step_video, interval=interval, blit=True)
        ani.save(f'./videos/{env.num_agents_max}UAVs_E_{i+1}_T_{current_time}.mp4', dpi=dpi, fps=fps, writer='ffmpeg')

    print(f"Average Episodic Reward: {episodic_reward_sum / num_episodes}")

    print("Done")

