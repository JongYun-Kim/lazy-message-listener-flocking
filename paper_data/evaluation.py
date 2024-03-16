import sys
import os
import pickle
import numpy as np

# add the path to the root directory of the project
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# load data with pickle
with open('/server/lazy-message-listener-flocking/data/2024-03-12_14-13-31/001_acs_vs_rl_seed_0-9_2024-03-12_14-13-31.pkl', 'rb') as f:
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
    
# # algorithm number
# 0: ACS
# 1: RL stochastic
# 2: RL deterministic

spatial_threshold = 60*0.7
velocity_threshold = 1.0
moving_average_window_size = 10
converged_window_size = 100

spatial_entropy = data['spatial_entropy'][:,:2,:] # Only ACS and RL stochastic
velocity_entropy = data['velocity_entropy'][:,:2,:] # Only ACS and RL stochastic

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


num_experiments, num_algos, _ = spatial_entropy.shape
convergence_counts = np.zeros((num_algos, 3))  # Additional column for simultaneous convergence
convergence_time_steps = {'spatial': [[] for _ in range(num_algos)],
                          'velocity': [[] for _ in range(num_algos)],
                          'simultaneous': [[] for _ in range(num_algos)]}
converged_experiments = {'spatial': [[] for _ in range(num_algos)],
                         'velocity': [[] for _ in range(num_algos)],
                         'simultaneous': [[] for _ in range(num_algos)]}
non_converged_experiments = {'spatial': [[] for _ in range(num_algos)],
                             'velocity': [[] for _ in range(num_algos)],
                             'simultaneous': [[] for _ in range(num_algos)]}
avg_diffs = {'spatial': [[] for _ in range(num_algos)],
             'velocity': [[] for _ in range(num_algos)]}  # Store average differences

for algo in range(num_algos):
    for experiment in range(num_experiments):
        spatial_ma = moving_average(spatial_entropy[experiment, algo], moving_average_window_size)
        velocity_ma = moving_average(velocity_entropy[experiment, algo], moving_average_window_size)

        spatial_converged_step, spatial_avg_diff = check_convergence_spatial(spatial_ma, spatial_threshold, converged_window_size)
        velocity_converged_step, velocity_avg_diff = check_convergence_velocity(velocity_ma, velocity_threshold, converged_window_size)

        if spatial_converged_step is not None:
            convergence_counts[algo, 0] += 1
            convergence_time_steps['spatial'][algo].append(spatial_converged_step + (converged_window_size - 1))
            converged_experiments['spatial'][algo].append(experiment)
            avg_diffs['spatial'][algo].append(spatial_avg_diff)
        else:
            non_converged_experiments['spatial'][algo].append(experiment)

        if velocity_converged_step is not None:
            convergence_counts[algo, 1] += 1
            convergence_time_steps['velocity'][algo].append(velocity_converged_step + (converged_window_size - 1))
            converged_experiments['velocity'][algo].append(experiment)
            avg_diffs['velocity'][algo].append(velocity_avg_diff)
        else:
            non_converged_experiments['velocity'][algo].append(experiment)

        # Check for simultaneous convergence
        if spatial_converged_step is not None and velocity_converged_step is not None:
            convergence_counts[algo, 2] += 1
            simultaneous_converged_step = max(spatial_converged_step, velocity_converged_step) + (converged_window_size - 1)
            convergence_time_steps['simultaneous'][algo].append(simultaneous_converged_step)
            converged_experiments['simultaneous'][algo].append(experiment)
        elif spatial_converged_step is not None or velocity_converged_step is not None:
            non_converged_experiments['simultaneous'][algo].append(experiment)

success_rates = convergence_counts / num_experiments

print("[[Success Rates]] (Spatial, Velocity, Simultaneous):")
for i, algo in enumerate(['ACS', 'RL Stochastic']):
    print(f"{algo}: Spatial: {success_rates[i, 0]:.2f}, Velocity: {success_rates[i, 1]:.2f}, Simultaneous: {success_rates[i, 2]:.2f}")

print("\n[[Average Converged Time Steps]]:")
for key in convergence_time_steps:
    avg_time_steps = [np.mean(steps) if steps else 'N/A' for steps in convergence_time_steps[key]]
    print(f"{key.capitalize()}: {' | '.join(str(step) for step in avg_time_steps)}")

print("\n[[Converged time steps for each experiment]]:")
for key in convergence_time_steps:
    print(f"{key.capitalize()}:")
    for i, algo in enumerate(['ACS', 'RL Stochastic']):
        experiments = convergence_time_steps[key][i]
        if experiments:
            print(f"  {algo}: {', '.join(map(str, experiments))}")
        else:
            print(f"  {algo}: None")

print("\n[[Average Differences at Convergence]]:")
for key in avg_diffs:
    avg_diffs_values = [np.mean(diffs) if diffs else 'N/A' for diffs in avg_diffs[key]]
    print(f"{key.capitalize()}: {' | '.join(str(diff) for diff in avg_diffs_values)}")

print("\n[[Average differences at convergence for each experiment]]:")
for key in avg_diffs:
    print(f"{key.capitalize()}:")
    for i, algo in enumerate(['ACS', 'RL Stochastic']):
        experiments = avg_diffs[key][i]
        if experiments:
            print(f"  {algo}: {', '.join(f'{experiment:.2f}' for experiment in experiments)}")
        else:
            print(f"  {algo}: None")

print("\n[[Detailed Converged Experiments Information]]:")
for key in converged_experiments:
    print(f"{key.capitalize()}:")
    for i, algo in enumerate(['ACS', 'RL Stochastic']):
        experiments = converged_experiments[key][i]
        if experiments:
            print(f"  {algo}: Experiments {', '.join(map(str, experiments))}")
        else:
            print(f"  {algo}: None")

print("\n[[Non-Converged Experiments Information]]:")
for key in non_converged_experiments:
    print(f"{key.capitalize()}:")
    for i, algo in enumerate(['ACS', 'RL Stochastic']):
        experiments = non_converged_experiments[key][i]
        if experiments:
            print(f"  {algo}: Experiments {', '.join(map(str, experiments))}")
        else:
            print(f"  {algo}: None")