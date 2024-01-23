import numpy as np
import ray
from env.envs import LazyMsgListenersEnv
import time

# Initialize Ray: use local mode for debugging
# ray.init(local_mode=True)
ray.init(num_cpus=15)

# Define a remote function for running a batch of episodes
@ray.remote
def run_episode_batch(env_config, actions, batch_size):
    batch_reward_sum = 0
    batch_episode_length = 0
    batch_loop_duration = 0

    for _ in range(batch_size):
        env = LazyMsgListenersEnv(env_config)
        done = False
        env.reset()
        reward_sum = 0
        start_time = time.time()

        while not done:
            obs, reward, done, info = env.step(actions)
            reward_sum += reward

        loop_duration = time.time() - start_time
        batch_reward_sum += reward_sum
        batch_episode_length += env.time_step
        batch_loop_duration += loop_duration

    return batch_reward_sum, batch_episode_length, batch_loop_duration

num_agents = 100
num_time_steps = 2000
config = {
    "num_agents_pool": [num_agents],
    "std_p_goal": 45,
    "max_time_steps": num_time_steps,
}
actions = LazyMsgListenersEnv(config).action_space.sample()
np.fill_diagonal(actions, 0)

# Run batches of episodes in parallel
num_episodes = 300
batch_size = 6  # Adjust batch size as needed
num_batches = num_episodes // batch_size
start_time = time.time()
futures = [run_episode_batch.remote(config, actions, batch_size) for _ in range(num_batches)]

# Gather results
results = ray.get(futures)
total_reward_sum = sum(result[0] for result in results)
total_episode_length = sum(result[1] for result in results)
total_loop_duration = sum(result[2] for result in results)

avg_episodic_reward = total_reward_sum / num_episodes
avg_episode_length = total_episode_length / num_episodes

print(f"Experiment Results in {num_agents} agents")
print("Average Episode Length: ", avg_episode_length)
print("Average Episodic Reward: ", avg_episodic_reward)
print(f"Total Loop Duration: {total_loop_duration} seconds")

duration = time.time() - start_time
print(f"Total Simulation Duration: {duration} seconds")

# Shut down Ray
ray.shutdown()

print("Done")