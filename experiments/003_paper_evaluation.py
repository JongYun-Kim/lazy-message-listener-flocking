import sys
import os
import pickle
import numpy as np

# file_path = './../data/2024-03-12_16-35-04/001_acs_vs_rl_seed_10-19_2024-03-12_16-35-04.pkl'
# file_path = './../data/2024-03-13_13-48-07/001_acs_vs_rl_seed_20-119_2024-03-13_13-48-07.pkl'
# file_path = './../data/2024-03-16_16-08-55/002_acs_rl_nature_seed_120-619_2024-03-16_16-08-55.pkl'
# file_path = './../data/2024-05-08_01-38-11/005_random_seed_120-619_2024-05-08_01-38-11.pkl'
# file_path = './../data/2024-05-08_09-27-47/005_random_seed_120-619_2024-05-08_09-27-47.pkl'
file_path = './../data/2024-05-16_06-38-33/006_heu_seed_120-319_2024-05-16_06-38-33.pkl'

# load data with pickle
with open(file_path, 'rb') as f:
    data = pickle.load(f)

# # data: dictionary [trajectories, velocities, spatial_entropy, velocity_entropy, actions, rewards, control, algo_str, seeds, originated_from]
# trajectories = (num_experiment, num_algos, max_time_steps, num_agents, 2)
# velocities = (num_experiment, num_algos, max_time_steps, num_agents, 2)
# spatial_entropy = (num_experiment, num_algos, max_time_steps)
# velocity_entropy = (num_experiment, num_algos, max_time_steps)
# actions = (num_experiment, num_algos, max_time_steps, num_agents, num_agents)
# rewards = (num_experiment, num_algos, max_time_steps)
# control = (num_experiment, num_algos, max_time_steps)
# seeds = (num_seeds,)

# Algorithm names
algo_str = data['algo_str']

spatial_threshold = 60 * np.sqrt(1/2)
velocity_threshold = 1.0
moving_average_window_size = 20
converged_window_size = 100

# Pick the data you want
spatial_entropy = data['spatial_entropy'][:, :, :]
velocity_entropy = data['velocity_entropy'][:, :, :]
assert len(algo_str) == spatial_entropy.shape[1] == velocity_entropy.shape[1]
num_agents = data['actions'].shape[-1]
sum_actions = np.sum(data['actions'], axis=(3, 4)) - num_agents  # Sum of actions (excluding self-loops)  # (num_experiment, num_algos, max_time_steps)


def moving_average(data, window_size=50):
    return np.convolve(data, np.ones(window_size) / window_size, mode='valid')


def check_convergence_spatial(moving_avg, threshold, window_size):
    for i in range(len(moving_avg) - window_size + 1):
        window = moving_avg[i:i + window_size]
        if np.all(window <= threshold):
            avg_difference = np.mean(threshold - window)  # Calculate average difference from threshold
            return i + window_size - 1, avg_difference
    return None, None


def check_convergence_velocity(moving_avg, threshold, window_size):
    for i in range(len(moving_avg) - window_size + 1):
        window = moving_avg[i:i + window_size]
        if np.all(window <= threshold):
            avg_difference = np.mean(window)  # Calculate average difference from threshold
            return i + window_size - 1, avg_difference
    return None, None


# Set up variables
num_experiments, num_algos, _ = spatial_entropy.shape
convergence_counts = np.zeros((num_algos, 3), dtype=np.float32)  # Additional column for simultaneous convergence
convergence_time_steps = {'spatial': [[] for _ in range(num_algos)],
                          'velocity': [[] for _ in range(num_algos)],
                          'simultaneous': [[] for _ in range(num_algos)]}
converged_experiments = {'spatial': [[] for _ in range(num_algos)],
                         'velocity': [[] for _ in range(num_algos)],
                         'simultaneous': [[] for _ in range(num_algos)]}
non_converged_experiments = {'spatial': [[] for _ in range(num_algos)],
                             'velocity': [[] for _ in range(num_algos)],
                             'simultaneous': [[] for _ in range(num_algos)]}
total_actions_at_convergence = {'simultaneous': [[] for _ in range(num_algos)]}
avg_diffs = {'spatial': [[] for _ in range(num_algos)],
             'velocity': [[] for _ in range(num_algos)]}  # Store average differences

# Check for convergence
for algo in range(num_algos):
    for experiment in range(num_experiments):
        spatial_ma = moving_average(spatial_entropy[experiment, algo], moving_average_window_size)
        velocity_ma = moving_average(velocity_entropy[experiment, algo], moving_average_window_size)

        spatial_converged_step, spatial_avg_diff = check_convergence_spatial(spatial_ma, spatial_threshold,
                                                                             converged_window_size)
        velocity_converged_step, velocity_avg_diff = check_convergence_velocity(velocity_ma, velocity_threshold,
                                                                                converged_window_size)

        if spatial_converged_step is not None:
            convergence_counts[algo, 0] += 1
            convergence_time_steps['spatial'][algo].append(spatial_converged_step + (moving_average_window_size - 1))
            converged_experiments['spatial'][algo].append(experiment)
            avg_diffs['spatial'][algo].append(spatial_avg_diff)
        else:
            non_converged_experiments['spatial'][algo].append(experiment)

        if velocity_converged_step is not None:
            convergence_counts[algo, 1] += 1
            convergence_time_steps['velocity'][algo].append(velocity_converged_step + (moving_average_window_size - 1))
            converged_experiments['velocity'][algo].append(experiment)
            avg_diffs['velocity'][algo].append(velocity_avg_diff)
        else:
            non_converged_experiments['velocity'][algo].append(experiment)

        # Check for simultaneous convergence and calculate total actions
        if spatial_converged_step is not None and velocity_converged_step is not None:
            convergence_counts[algo, 2] += 1
            simultaneous_converged_step = max(spatial_converged_step, velocity_converged_step) + (
                        moving_average_window_size - 1)
            convergence_time_steps['simultaneous'][algo].append(simultaneous_converged_step)
            converged_experiments['simultaneous'][algo].append(experiment)
            # Sum actions up to simultaneous convergence step for the specific algorithm
            total_actions = np.sum(sum_actions[experiment, algo, :simultaneous_converged_step + 1])
            total_actions_at_convergence['simultaneous'][algo].append(total_actions)
        elif spatial_converged_step is not None or velocity_converged_step is not None:
            non_converged_experiments['simultaneous'][algo].append(experiment)

success_rates = convergence_counts / num_experiments

# Print results
print("\n[[Success Rates]] (Spatial, Velocity, Simultaneous):")
for i, algo in enumerate(algo_str):
    print(
        f"{algo}: Spatial: {success_rates[i, 0]:.2f}, Velocity: {success_rates[i, 1]:.2f}, Simultaneous: {success_rates[i, 2]:.2f}")

print("\n[[Average Converged Time Steps]]:")
for key in convergence_time_steps:
    stats = [f"{np.mean(steps):.2f} (±{np.std(steps):.2f})" if steps else 'N/A' for steps in convergence_time_steps[key]]
    print(f"{key.capitalize()}: {' | '.join(stats)}")

print("\n[[Converged time steps for each experiment]]:")
for key in convergence_time_steps:
    print(f"{key.capitalize()}:")
    for i, algo in enumerate(algo_str):
        experiments = convergence_time_steps[key][i]
        if experiments:
            print(f"  {algo}: {', '.join(map(str, experiments))}")
        else:
            print(f"  {algo}: None")

# Print total actions at convergence
print("\n[[Total Neighbor Info Used at Simultaneous Convergence]]:")
for i, algo in enumerate(algo_str):
    actions = total_actions_at_convergence['simultaneous'][i]
    if actions:
        print(f"  {algo}: {', '.join(map(str, actions))}")
    else:
        print(f"  {algo}: None")

# Print average actions at convergence
print("\n[[Average Actions at Convergence]]:")
for i, algo in enumerate(algo_str):
    algo_actions = total_actions_at_convergence['simultaneous'][i]
    if algo_actions:
        print(f"  {algo}: {np.mean(algo_actions):.2f}, (±{np.std(algo_actions):.2f})")
    else:
        print(f"  {algo}: None")

# Print average actions per convergence time steps
print("\n[[Average Actions per Convergence Time Steps -- Information efficiency]]:")
asdf=0
for i, algo in enumerate(algo_str):
    actions_list = total_actions_at_convergence['simultaneous'][i]
    steps_list = convergence_time_steps['simultaneous'][i]
    if steps_list:
        info_eff = np.mean((1000- np.array(steps_list)) / np.array(actions_list))  # 1000 is the max time steps
        info_eff_log = np.mean((1000- np.array(steps_list)) / np.log2(np.array(actions_list)))
        conversion_rate = 500 / len(actions_list)
        info_eff = info_eff * conversion_rate
        info_eff_log = info_eff_log * conversion_rate
        print(f"  {algo}: {info_eff:.2f}")
        print(f"  {algo}:             {info_eff_log:.2f}")
        asdf += 1
    else:
        print(f"  {algo}: 0")

print("\n[[Average Differences at Convergence]]:")
for key in avg_diffs:
    avg_diffs_values = [np.mean(diffs) if diffs else 'N/A' for diffs in avg_diffs[key]]
    print(f"{key.capitalize()}: {' | '.join(str(diff) for diff in avg_diffs_values)}")

print("\n[[Average differences at convergence for each experiment]]:")
for key in avg_diffs:
    print(f"{key.capitalize()}:")
    for i, algo in enumerate(algo_str):
        experiments = avg_diffs[key][i]
        if experiments:
            print(f"  {algo}: {', '.join(f'{experiment:.2f}' for experiment in experiments)}")
        else:
            print(f"  {algo}: None")

print("\n[[Detailed Converged Experiments Information]]:")
for key in converged_experiments:
    print(f"{key.capitalize()}:")
    for i, algo in enumerate(algo_str):
        experiments = converged_experiments[key][i]
        if experiments:
            print(f"  {algo}: Experiments {', '.join(map(str, experiments))}")
        else:
            print(f"  {algo}: None")

print("\n[[Non-Converged Experiments Information]]:")
for key in non_converged_experiments:
    print(f"{key.capitalize()}:")
    for i, algo in enumerate(algo_str):
        experiments = non_converged_experiments[key][i]
        if experiments:
            print(f"  {algo}: Experiments {', '.join(map(str, experiments))}")
        else:
            print(f"  {algo}: None")

print("Done!")

