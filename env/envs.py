# It is what it is...
import gym
from gym.utils import seeding
from gym.spaces import Box, Discrete, Dict, MultiDiscrete, MultiBinary
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from ray.rllib.utils.typing import (
    AgentID,
    # EnvCreator,
    # EnvID,
    # EnvType,
    MultiAgentDict,
    # MultiEnvDict,
)

matplotlib.use('TkAgg')  # To avoid the MacOS backend; but just follow your needs

# Compatibility layer for np.bool_ and np.bool
if not hasattr(np, 'bool_'):
    np.bool_ = np.bool


class LazyMsgListenersEnv(gym.Env):
    """
    *Lazy Message-Listeners Environment*
    - Description: augmented C-S flocking control runs based on action from the model
    - Action:
    - Reward:
    - Observation:
    - State:
      - agent_states[i] = [x, y, vx, vy, theta, (u)], neighbor_masks, padding_mask
    - Episode Termination:

    """

    def __init__(self, config):
        super().__init__()  # Init gym.Env first
        self.seed()  # Seed control

        # Get the config
        self.config = config
        # observation_dim: dimension of the agent-wise observations
        self.obs_dim = self.config["obs_dim"] if "obs_dim" in self.config else 4
        # env_mode: singe agent (ray) env or multi-agent (ray) env
        self.env_mode = self.config["env_mode"] if "env_mode" in self.config else "single_env"
        # action_type: binary_vector or else
        # # binary_vector: default; action is a binary vector(s) of shape (num_agents_max, num_agents_max)
        # # radius: action is a radius of the communication range; shape (1,) (i.e. scalar)
        # # continuous_vector: action is a continuous vector(s) of shape (num_agents_max, num_agents_max)
        self.action_type = self.config["action_type"] if "action_type" in self.config else "binary_vector"
        # num_agents_pool: pool of possible num_agents;
        # # type: <tuple> for range
        # # type: <ndarray> or python <list> of integers for a manual list pool of num_agents
        # # type: now, we accept <int> for a single num_agents
        # # note: self.num_agents_pool is transformed to an ndarray in _validate_config()
        # # NO DEFAULT HERE; MUST BE SPECIFIED
        self.num_agents_pool = self.config["num_agents_pool"]
        # agent_name_suffix: suffix of the agent name
        self.agent_name_prefix = self.config["agent_name_prefix"] if "agent_name_prefix" in self.config else "agent_"
        # control_config: config for the C-S flocking control algorithm
        if "control_config" in self.config:
            self.control_config = self.get_default_control_config(self.config["control_config"])
        else:
            self.control_config = self.get_default_control_config()
        self.dt = self.control_config["dt"] if "dt" in self.control_config else 0.1  # time step size
        # comm_range: communication range in meters; None means FULLY CONNECTED !!
        self.comm_range = self.config["comm_range"] if "comm_range" in self.config else None
        # termination conditions (max_time_step, std_p_goal, std_v_goal, std_p_rate_goat, std_v_rate_goal)
        self.max_time_steps = self.config["max_time_steps"] if "max_time_steps" in self.config else 1000
        self.std_p_goal = self.control_config["std_p_goal"] if "std_p_goal" in self.control_config else 0.7 * self.control_config["predefined_distance"]
        self.std_v_goal = self.control_config["std_v_goal"] if "std_v_goal" in self.control_config else 0.1
        self.std_p_rate_goal = self.control_config["std_p_rate_goal"] if "std_p_rate_goal" in self.control_config else 0.1
        self.std_v_rate_goal = self.control_config["std_v_rate_goal"] if "std_v_rate_goal" in self.control_config else 0.2
        # use_fixed_episode_length: may need it in training
        self.use_fixed_episode_length = self.config["use_fixed_episode_length"] if "use_fixed_episode_length" in self.config else False
        # get_state_hist
        self.get_state_hist = self.config["get_state_hist"] if "get_state_hist" in self.config else False
        self.get_action_hist = self.config["get_action_hist"] if "get_action_hist" in self.config else False

        # Other attributes
        # # Number of agents
        self.num_agents = None      # defined in reset()
        self.num_agents_min = None  # defined in _validate_config()
        self.num_agents_max = None  # defined in _validate_config()

        # # States
        # # # state: dict := {"agent_states":   ndarray,  # shape (num_agents_max, data_dim); absolute states!!
        #                                        [x, y, vx, vy, theta]; absolute states!!
        #                     "neighbor_masks": ndarray,  # shape (num_agents_max, num_agents_max)
        #                                        1 if neighbor, 0 if not;  self loop is 0
        #                     "padding_mask":   ndarray,  # shape (num_agents_max)
        #                                        1 if agent,    0 if padding
        self.state = None
        # # # rel_state: dict := {"rel_agent_positions": ndarray,   # shape (num_agents_max, num_agents_max, 2)
        #                         "rel_agent_velocities": ndarray,  # shape (num_agents_max, num_agents_max, 2)
        #                         "rel_agent_headings": ndarray,    # shape (num_agents_max, num_agents_max)  # 2-D !!!
        #                         "rel_agent_dists": ndarray        # shape (num_agents_max, num_agents_max)
        #                         }
        #  }
        self.rel_state = None
        self.initial_state = None
        self.agent_states_hist = None
        self.neighbor_masks_hist = None
        # self.padding_mask_hist = None
        self.action_hist = None
        self.has_lost_comm = None
        # # Standard deviations
        self.std_pos_hist = None
        self.std_vel_hist = None
        # # Time steps
        self.time_step = None
        # self.agent_time_step = None

        # # Validate the config
        self._validate_config()

        # Define ACTION SPACE
        self.action_dtype = None
        if self.env_mode == "single_env":
            if self.action_type == "binary_vector":
                # self.action_space = Box(low=0, high=1, shape=(self.num_agents_max, self.num_agents_max), dtype=np.bool_)
                self.action_space = Box(low=0, high=1, shape=(self.num_agents_max, self.num_agents_max), dtype=np.int8)
                # self.action_space = MultiBinary((self.num_agents_max, self.num_agents_max))
                self.action_dtype = np.int8
            elif self.action_type == "radius":
                self.action_space = Box(low=0, high=1, shape=self.num_agents_max, dtype=np.float64)
                self.action_dtype = np.float64
            elif self.action_type == "continuous_vector":
                self.action_space = Box(low=0, high=1, shape=(self.num_agents_max, self.num_agents_max), dtype=np.float64)
                self.action_dtype = np.float64
            else:
                raise NotImplementedError("action_type must be either binary_vector or radius or continuous_vector")
        elif self.env_mode == "multi_env":
            print("WARNING (env.__init__): multi_env is experimental; not fully implemented yet")
            if self.action_type == "binary_vector":
                self.action_space = Dict({
                    self.agent_name_prefix + str(i): Box(low=0, high=1, shape=(self.num_agents_max,), dtype=np.bool_)
                    for i in range(self.num_agents_max)
                })
            elif self.action_type == "radius":
                self.action_space = Dict({
                    self.agent_name_prefix + str(i): Box(low=0, high=1, shape=(1,), dtype=np.float64)
                    for i in range(self.num_agents_max)
                })
            elif self.action_type == "continuous_vector":
                self.action_space = Dict({
                    self.agent_name_prefix + str(i): Box(low=0, high=1, shape=(self.num_agents_max,), dtype=np.float64)
                    for i in range(self.num_agents_max)
                })
            else:
                raise NotImplementedError("action_type must be either binary_vector or radius or continuous_vector")
        else:
            raise NotImplementedError("env_mode must be either single_env or multi_env")

        # Define OBSERVATION SPACE
        if self.env_mode == "single_env":
            self.observation_space = Dict({
                "centralized_agents_info": Box(low=-np.inf, high=np.inf, shape=(self.num_agents_max, self.obs_dim), dtype=np.float64),
                "neighbor_masks": Box(low=0, high=1, shape=(self.num_agents_max, self.num_agents_max), dtype=np.bool_),
                "padding_mask": Box(low=0, high=1, shape=(self.num_agents_max,), dtype=np.bool_)
            })
        elif self.env_mode == "multi_env":
            self.observation_space = Dict({
                self.agent_name_prefix + str(i): Dict({
                    "centralized_agent_info": Box(low=-np.inf, high=np.inf, shape=(5,), dtype=np.float64),
                    "neighbor_mask": Box(low=0, high=1, shape=(self.num_agents_max,), dtype=np.bool_),
                    "padding_mask": Box(low=0, high=1, shape=(self.num_agents_max,), dtype=np.bool_)
                }) for i in range(self.num_agents_max)
            })

    def _validate_config(self):
        # env_mode: must be either "single_env" or "multi_env"
        assert self.env_mode in ["single_env", "multi_env"], "env_mode must be either single_env or multi_env"

        # num_agents_pool: must be a tuple(range)/ndarray of int-s (list is also okay instead of ndarray for list-pool)
        if isinstance(self.num_agents_pool, int):
            assert self.num_agents_pool > 1, "num_agents_pool must be > 1"
            self.num_agents_pool = np.array([self.num_agents_pool])
        assert isinstance(self.num_agents_pool, (tuple, np.ndarray, list)), "num_agents_pool must be a tuple or ndarray"
        assert all(np.issubdtype(type(x), int) for x in self.num_agents_pool), "all values in num_agents_pool must be int-s"
        if isinstance(self.num_agents_pool, list):
            self.num_agents_pool = np.array(self.num_agents_pool)  # convert to np-array
        if isinstance(self.num_agents_pool, tuple):
            assert len(self.num_agents_pool) == 2, "num_agents_pool must be a tuple of length 2, as (min, max); a range"
            assert self.num_agents_pool[0] <= self.num_agents_pool[1], "min of num_agents_pool must be <= max"
            assert self.num_agents_pool[0] > 1, "min of num_agents_pool must be > 1"
            self.num_agents_pool = np.arange(self.num_agents_pool[0], self.num_agents_pool[1] + 1)
        elif isinstance(self.num_agents_pool, np.ndarray):
            assert self.num_agents_pool.size > 0, "num_agents_pool must not be empty"
            assert len(self.num_agents_pool.shape) == 1, "num_agents_pool must be a np-array of shape (n, ), n > 1"
            assert all(self.num_agents_pool > 1), "all values in num_agents_pool must be > 1"
        else:
            raise NotImplementedError("Something wrong; check _validate_config() of LazyFusionEnv; must not reach here")
        # Note: Now self.num_agents_pool is a ndarray of possible num_agents; ㅇㅋ?

        # Set num_agents_min and num_agents_max
        self.num_agents_min = self.num_agents_pool.min()
        self.num_agents_max = self.num_agents_pool.max()

        # max_time_step: must be an int and > 0
        assert isinstance(self.max_time_steps, int), "max_time_step must be an int"
        assert self.max_time_steps > 0, "max_time_step must be > 0"

        # variable: condition

    def get_default_control_config(self, control_config=None):
        """
        Set the default config of your control algorithm; here, C-S flocking control
        If you use another control algorithm, override this method, bro.
        :return: default_control_config
        """
        # CS params:
        # Set the default control config
        default_control_config = {
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
        }
        if control_config is not None:
            # Check if the control_config has all the keys
            assert all(key in default_control_config for key in control_config.keys()), \
                "control_config must have all the keys of default_control_config"
            # Update the default_control_config with the control_config
            default_control_config.update(control_config)

        return default_control_config

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def custom_reset(self, p_, v_, th_, num_agents_max=None, comm_range=None):
        """
        Custom reset method to reset the environment with the given initial states
        :param p_: ndarray of shape (num_agents, 2)
        :param v_: ndarray of shape (num_agents, 2)
        :param th_: ndarray of shape (num_agents, )
        :param comm_range: float; communication range
        :param num_agents_max: int; maximum number of agents
        :return: obs
        """
        # Dummy reset
        self.reset()
        # Init time steps
        self.time_step = 0
        # self.agent_time_step = np.zeros(self.num_agents_max, dtype=np.int32)

        # Get initial num_agents
        num_agents = len(p_)
        self.num_agents = num_agents
        num_agents_max = num_agents if num_agents_max is None else num_agents_max

        # Check dimension
        assert p_.shape[0] == v_.shape[0] == th_.shape[0], "p_, v_, th_ must have the same shape[0]"
        assert self.num_agents_min <= num_agents <= self.num_agents_max, "num_agents_max must be <= self.num_agents_max"
        assert num_agents_max == self.num_agents_max, "num_agents_max must be == self.num_agents_max"
        assert p_.shape[1] == v_.shape[1] == 2, "p_, v_ must have shape[1] == 2"
        assert th_.shape[1] == 1, "th_ must have shape[1] == 1"

        # Get initial agent states
        # # agent_states: [x, y, vx, vy, theta]
        p = np.zeros((num_agents_max, 2), dtype=np.float64)  # (num_agents_max, 2)
        p[:num_agents, :] = p_
        v = np.zeros((num_agents_max, 2), dtype=np.float64)  # (num_agents_max, 2)
        v[:num_agents, :] = v_
        th = np.zeros(num_agents_max, dtype=np.float64)  # (num_agents_max, )
        th[:num_agents] = th_
        # Concatenate p v th
        agent_states = np.concatenate([p, v, th[:, np.newaxis]], axis=1)  # (num_agents_max, 5)
        # # padding_mask
        padding_mask = np.zeros(num_agents_max, dtype=np.bool_)  # (num_agents_max, )
        padding_mask[:num_agents] = True
        # # neighbor_masks
        self.comm_range = comm_range
        if self.comm_range is None:
            neighbor_masks = np.ones((num_agents_max, num_agents_max), dtype=np.bool_)
        else:
            neighbor_masks = self.compute_neighbor_agents(
                agent_states=agent_states, padding_mask=padding_mask, communication_range=self.comm_range)[0]
        # # state!
        self.state = {"agent_states": agent_states, "neighbor_masks": neighbor_masks, "padding_mask": padding_mask}
        self.initial_state = self.state
        self.has_lost_comm = False

        # Get relative state
        self.rel_state = self.get_relative_state(state=self.state)

        # Get obs
        obs = self.get_obs(state=self.state, rel_state=self.rel_state, control_inputs=np.zeros(num_agents_max))

        return obs

    def reset(self):
        # Init time steps
        self.time_step = 0
        # self.agent_time_step = np.zeros(self.num_agents_max, dtype=np.int32)

        # Get initial num_agents
        self.num_agents = np.random.choice(self.num_agents_pool)      # randomly choose the num_agents
        padding_mask = np.zeros(self.num_agents_max, dtype=np.bool_)  # (num_agents_max, )
        padding_mask[:self.num_agents] = True

        # Init the state: agent_states [x,y,vx,vy,theta], neighbor_masks[T/F (n,n)], padding_mask[T/F (n)]
        # # Generate initial agent states
        p = np.zeros((self.num_agents_max, 2), dtype=np.float64)  # (num_agents_max, 2)
        l2 = self.control_config["initial_position_bound"]/2
        p[:self.num_agents, :] = np.random.uniform(-l2, l2, size=(self.num_agents, 2))
        th = np.zeros(self.num_agents_max, dtype=np.float64)  # (num_agents_max, )
        th[:self.num_agents] = np.random.uniform(-np.pi, np.pi, size=(self.num_agents,))
        v = self.control_config["speed"] * np.stack([np.cos(th), np.sin(th)], axis=1)
        v[self.num_agents:] = 0
        # # Concatenate p v th
        agent_states = np.concatenate([p, v, th[:, np.newaxis]], axis=1)  # (num_agents_max, 5)
        if self.comm_range is None:
            neighbor_masks = np.ones((self.num_agents_max, self.num_agents_max), dtype=np.bool_)
        else:
            neighbor_masks = self.compute_neighbor_agents(
                agent_states=agent_states, padding_mask=padding_mask, communication_range=self.comm_range)[0]
        self.state = {"agent_states": agent_states, "neighbor_masks": neighbor_masks, "padding_mask": padding_mask}
        self.has_lost_comm = False

        # Get relative state
        self.rel_state = self.get_relative_state(state=self.state)

        # Get obs
        obs = self.get_obs(state=self.state, rel_state=self.rel_state, control_inputs=np.zeros(self.num_agents_max))

        # Std
        self.std_pos_hist = np.zeros(self.max_time_steps)
        self.std_vel_hist = np.zeros(self.max_time_steps)

        # Other settings
        if self.get_state_hist:
            self.agent_states_hist = np.zeros((self.max_time_steps, self.num_agents_max, 5))
            self.neighbor_masks_hist = np.zeros((self.max_time_steps, self.num_agents_max, self.num_agents_max))
            self.initial_state = self.state
        if self.get_action_hist:
            self.action_hist = np.zeros((self.max_time_steps, self.num_agents_max, self.num_agents_max), dtype=np.bool_)

        return obs

    def step(self, action):
        """
        Step the environment
        :param action: your_model_output; ndarray of shape (num_agents_max, num_agents_max) expected under the default
        :return: obs, reward, done, info
        """
        state = self.state  # state of the class (flock);
        rel_state = self.rel_state  # did NOT consider the communication network, DELIBERATELY

        # Interpret the action (i.e. model output)
        action_interpreted = self.interpret_action(model_output=action)  # TODO: consider radius action type, *L8R*
        joint_action = self.multi_to_single(action_interpreted) if self.env_mode == "multi_env" else action_interpreted
        joint_action = self.to_binary_action(joint_action)  # (num_agents_max, num_agents_max)
        self.validate_action(action=joint_action,
                             neighbor_masks=state["neighbor_masks"], padding_mask=state["padding_mask"])

        # Step the environment in *single agent* setting!, which may be faster due to vectorization-like things
        # # s` = T(s, a)
        next_state, control_inputs, comm_loss_agents = self.env_transition(state, rel_state, joint_action)
        next_rel_state = self.get_relative_state(state=next_state)
        # # r = R(s, a, s`)
        rewards = self._compute_rewards(
            state=state, action=joint_action, next_state=next_state, control_inputs=control_inputs)
        # # o = H(s`)
        obs = self.get_obs(state=next_state, rel_state=next_rel_state, control_inputs=control_inputs)

        # Check episode termination
        done = self.check_episode_termination(state=next_state, rel_state=next_rel_state, comm_loss_agents=comm_loss_agents)

        # Get custom reward if implemented
        custom_reward = self.compute_custom_reward(state, rel_state, control_inputs, rewards, done)
        _reward = rewards.sum() / self.num_agents if self.env_mode == "single_env" else self.single_to_multi(rewards)
        reward = custom_reward if custom_reward is not NotImplemented else _reward

        # Collect info
        info = {
            "std_pos": self.std_pos_hist[self.time_step],
            "std_vel": self.std_vel_hist[self.time_step],
            "original_rewards": _reward,
            "comm_loss_agents": comm_loss_agents,
        }
        info = self.get_extra_info(info, next_state, next_rel_state, control_inputs, rewards, done)
        if self.get_state_hist:
            self.agent_states_hist[self.time_step] = next_state["agent_states"]
            self.neighbor_masks_hist[self.time_step] = next_state["neighbor_masks"]
        if self.get_action_hist:
            self.action_hist[self.time_step] = joint_action

        # Update self.state and the self.rel_state
        self.state = next_state
        self.rel_state = next_rel_state
        # Update time steps
        self.time_step += 1
        # self.agent_time_step[state["padding_mask"]] += 1
        return obs, reward, done, info

    def get_relative_state(self, state):
        """
        Get the relative state (positions, velocities, headings, distances) from the absolute state
        """
        agent_positions = state["agent_states"][:, :2]
        agent_velocities = state["agent_states"][:, 2:4]
        agent_headings = state["agent_states"][:, 4, np.newaxis]  # shape (num_agents_max, 1): 2-D array
        # neighbor_masks = state["neighbor_masks"]  # shape (num_agents_max, num_agents_max)
        padding_mask = state["padding_mask"]    # shape (num_agents_max)

        # Get relative positions and distances
        rel_agent_positions, rel_agent_dists = self.get_relative_info(
            data=agent_positions, mask=padding_mask, get_dist=True, get_active_only=False)

        # Get relative velocities
        rel_agent_velocities, _ = self.get_relative_info(
            data=agent_velocities, mask=padding_mask, get_dist=False, get_active_only=False)

        # Get relative headings
        _, rel_agent_headings = self.get_relative_info(
            data=agent_headings, mask=padding_mask, get_dist=True, get_active_only=False)

        # rel_state: dict
        rel_state = {"rel_agent_positions": rel_agent_positions,
                     "rel_agent_velocities": rel_agent_velocities,
                     "rel_agent_headings": rel_agent_headings,
                     "rel_agent_dists": rel_agent_dists
                     }

        return rel_state

    def interpret_action(self, model_output):
        """
        Please implement this method as you need. Currently, it just passes the model_output.
        Interprets the model output
        :param model_output
        :return: interpreted_action
        """
        return model_output

    def validate_action(self, action, neighbor_masks, padding_mask):
        """
        Validates the action by checking the neighbor_mask and padding_mask
        :param action:  (num_agents_max, num_agents_max)
        :param neighbor_masks: (num_agents_max, num_agents_max)
        :param padding_mask: (num_agents_max)
        :return: None
        """
        # Check the dtype and shape of the action
        if self.action_type == "binary_vector":  # TODO 땜빵
            if action.dtype == np.int64:
                self.action_dtype = np.int64
        assert action.dtype == self.action_dtype, "action must be a(an) {} ndarray".format(self.action_dtype)
        assert action.shape == (self.num_agents_max, self.num_agents_max), \
            "action must be a boolean ndarray of shape (num_agents_max, num_agents_max)"

        # Ensure the diagonal elements are all ones (all with self-loops); if not, set them to ones
        if not np.all(np.diag(action) == 1):
            np.fill_diagonal(action, 1)  # Directly modifies the action array to set diagonal elements to 1
            print("WARNING (env.validate_action): diag(action) not all 1; Self-loops fixed in 'action'.")

        # Check action value based on the neighbor_mask and padding_mask
        # Note: your model might output a masked action. If that's not available, you can ignore this part.
        assert np.all((neighbor_masks | ~action)), "action[i, j] == 1 must not found if neighbor_mask[i, j] == 0"
        assert np.all((padding_mask[:, None] | ~action)), "action[i, j] == 1 must not found if padding_mask[j] == 0"

        # # Efficiently check for rows with all zeros (excluding self-loops)
        # if self.time_step != 0:
        #     assert np.all(action.sum(axis=1) - np.diag(action) > 0), \
        #         "Each row in action, except self-loops, must have at least one True value"

    def to_binary_action(self, action_in_another_type):
        if self.action_type == "binary_vector":
            return action_in_another_type
        elif self.action_type == "radius":
            # action_in_another_type: ndarray of shape (num_agents_max, )
            # # action_in_another_type[i] is the radius of the communication range of agent i

            # Check if the radius is positive and less than the communication range (non-padded agents only)
            assert np.all(action_in_another_type[self.state["padding_mask"]] > 0), \
                "action_in_another_type[i] must be > 0 for all non-padded agents"
            assert np.all(action_in_another_type[self.state["padding_mask"]] <= 1), \
                "action_in_another_type[i] must be <= self.comm_range for all non-padded agents"

            # Set the action
            agent_wise_comm_range = self.comm_range * action_in_another_type[self.state["padding_mask"], np.newaxis]  # (num_agents, 1)
            action_in_binary = self.compute_neighbor_agents(
                agent_states=self.state["agent_states"], padding_mask=self.state["padding_mask"],
                communication_range=agent_wise_comm_range)[0]  # TODO: Check loss of communication
            return action_in_binary  # (num_agents_max, num_agents_max)
        elif self.action_type == "continuous_vector":
            raise NotImplementedError("continuous_vector action_type is not implemented yet")
        return None

    def multi_to_single(self, variable_in_multi: MultiAgentDict):
        """
        Converts a multi-agent variable to a single-agent variable
        Assumption: homogeneous agents
        :param variable_in_multi: dict {agent_name_suffix + str(i): variable_in_single[i]}; {str: ndarray}
        :return: variable_in_single: ndarray of shape (num_agents, data...)
        """
        # Add extra dimension of each agent's variable on axis=0
        assert variable_in_multi[self.agent_name_prefix + str(0)].shape[0] == self.num_agents, \
            "num_agents must == variable_in_multi['agent_0'].shape[0]"
        variable_in_single = np.array(variable_in_multi.values())  # (num_agents, ...)

        return variable_in_single

    def single_to_multi(self, variable_in_single: np.ndarray):
        """
        Converts a single-agent variable to a multi-agent variable
        Assumption: homogeneous agents
        :param variable_in_single: ndarray of shape (num_agents, data...)
        :return: variable_in_multi
        """
        # Remove the extra dimension of each agent's variable on axis=0 and use self.agent_name_suffix with i as keys
        variable_in_multi = {}
        assert variable_in_single.shape[0] == self.num_agents_max, "variable_in_single[0] must be self.num_agents_max"
        for i in range(self.num_agents_max):
            variable_in_multi[self.agent_name_prefix + str(i)] = variable_in_single[i]

        return variable_in_multi

    def env_transition(self, state, rel_state, action):
        """
        Transition the environment; all args in single-rl-agent settings
        s` = T(s, a); deterministic
        :param state: dict:
        :param rel_state: dict:
        :param action: ndarray of shape (num_agents_max, num_agents_max)
        :return: next_state: dict; control_inputs: (num_agents_max, )
        """
        # Validate the laziness_vectors
        # self.validate_action(action=action, neighbor_masks=state["neighbor_masks"], padding_mask=state["padding_mask"])

        # 0. Apply lazy message actions: alters the neighbor_masks!
        lazy_listening_msg_masks = np.logical_and(state["neighbor_masks"], action)  # (num_agents_max, num_agents_max)

        # 1. Get control inputs based on the flocking control algorithm with the lazy listener's network
        control_inputs = self.get_control_inputs(state, rel_state, lazy_listening_msg_masks)  # (num_agents_max, )

        # 2. Update the agent states based on the control inputs
        next_agent_states = self.update_agent_states(state=state, control_inputs=control_inputs)

        # 3. Update network topology (i.e. neighbor_masks) based on the new agent states
        if self.comm_range is None:
            next_neighbor_masks = state["neighbor_masks"]
            comm_loss_agents = None
        else:
            next_neighbor_masks, comm_loss_agents = self.compute_neighbor_agents(
                agent_states=next_agent_states, padding_mask=state["padding_mask"], communication_range=self.comm_range)

        # 4. Update the active agents (i.e. padding_mask); you may lose or gain agents
        # next_padding_mask = self.update_active_agents(
        #     agent_states=next_agent_states, padding_mask=state["padding_mask"], communication_range=self.comm_range)
        # self.num_agents = next_padding_mask.sum()  # update the number of agents

        # 5. Update the state
        next_state = {"agent_states": next_agent_states,
                      "neighbor_masks": next_neighbor_masks,
                      "padding_mask": state["padding_mask"]
                      }

        return next_state, control_inputs, comm_loss_agents

    def get_control_inputs(self, state, rel_state, new_network):
        """
        Get the control inputs based on the agent states
        :return: control_inputs (num_agents_max)
        """
        # Please Work with Active Agents Only

        # Get rel_pos, rel_dist, rel_vel, rel_ang, abs_ang, padding_mask, neighbor_masks
        rel_pos = rel_state["rel_agent_positions"]   # (num_agents_max, num_agents_max, 2)
        rel_dist = rel_state["rel_agent_dists"]      # (num_agents_max, num_agents_max)
        rel_vel = rel_state["rel_agent_velocities"]  # (num_agents_max, num_agents_max, 2)
        rel_ang = rel_state["rel_agent_headings"]    # (num_agents_max, num_agents_max)
        abs_ang = state["agent_states"][:, 4]        # (num_agents_max, )
        padding_mask = state["padding_mask"]         # (num_agents_max)
        neighbor_masks = new_network  # (num_agents_max, num_agents_max)

        # Get data of the active agents
        active_agents_indices = np.nonzero(padding_mask)[0]  # (num_agents, )
        active_agents_indices_2d = np.ix_(active_agents_indices, active_agents_indices)  # (num_agents,num_agents)
        p = rel_pos[active_agents_indices_2d]  # (num_agents, num_agents, 2)
        r = rel_dist[active_agents_indices_2d] + (np.eye(self.num_agents) * np.finfo(float).eps)  # (num_agents, num_agents)
        v = rel_vel[active_agents_indices_2d]  # (num_agents, num_agents, 2)
        th = rel_ang[active_agents_indices_2d]  # (num_agents, num_agents)
        th_i = abs_ang[padding_mask]  # (num_agents, )
        net = neighbor_masks[active_agents_indices_2d]  # (num_agents, num_agents) may be no self-loops (i.e. 0 on diag)
        N = (net + (np.eye(self.num_agents) * np.finfo(float).eps)).sum(axis=1)  # (num_agents, )

        # Get control config
        beta = self.control_config["communication_decay_rate"]
        lam = self.control_config["inter_agent_strength"]
        k1 = self.control_config["k1"]
        k2 = self.control_config["k2"]
        spd = self.control_config["speed"]
        u_max = self.control_config["max_turn_rate"]
        r0 = self.control_config["predefined_distance"]
        sig = self.control_config["bonding_strength"]

        # 1. Compute Alignment Control Input
        # # u_cs = (lambda/n(N_i)) * sum_{j in N_i}[ psi(r_ij)sin(θ_j - θ_i) ],
        # # where N_i is the set of neighbors of agent i,
        # # psi(r_ij) = 1/(1+r_ij^2)^(beta),
        # # r_ij = ||X_j - X_i||, X_i = (x_i, y_i),
        psi = (1 + r**2)**(-beta)  # (num_agents, num_agents)
        alignment_error = np.sin(th)  # (num_agents, num_agents)
        u_cs = (lam / N) * (psi * alignment_error * net).sum(axis=1)  # (num_agents, )

        # 2. Compute Cohesion and Separation Control Input
        # # u_coh[i] = (sigma/N*V)
        # #            * sum_(j in N_i)
        # #               [
        # #                   {
        # #                       (K1/(2*r_ij^2))*<-rel_vel, -rel_pos> + (K2/(2*r_ij^2))*(r_ij-R)
        # #                   }
        # #                   * <[-sin(θ_i), cos(θ_i)]^T, rel_pos>
        # #               ]
        # # where N_i is the set of neighbors of agent i,
        # # r_ij = ||X_j - X_i||, X_i = (x_i, y_i),
        # # rel_vel = (vx_j - vx_i, vy_j - vy_i),
        # # rel_pos = (x_j - x_i, y_j - y_i),
        sig_NV = sig / (N * spd)  # (num_agents, )
        k1_2r2 = k1 / (2 * r**2)  # (num_agents, num_agents)
        k2_2r = k2 / (2 * r)  # (num_agents, num_agents)
        v_dot_p = np.einsum('ijk,ijk->ij', v, p)  # (num_agents, num_agents)
        r_minus_r0 = r - r0  # (num_agents, num_agents)
        # below dir_vec and dir_dot_p in the commented lines are the old way of computing the dot product
        # dir_vec = np.stack([-np.sin(th_i), np.cos(th_i)], axis=1)  # (num_agents, 2)
        # dir_vec = np.tile(dir_vec[:, np.newaxis, :], (1, self.num_agents, 1))  # (num_agents, num_agents, 2)
        # dir_dot_p = np.einsum('ijk,ijk->ij', dir_vec, p)  # (num_agents, num_agents)
        sin_th_i = -np.sin(th_i)  # (num_agents, )
        cos_th_i = np.cos(th_i)   # (num_agents, )
        dir_dot_p = sin_th_i[:, np.newaxis]*p[:, :, 0] + cos_th_i[:, np.newaxis]*p[:, :, 1]  # (num_agents, num_agents)
        u_coh = sig_NV * np.sum((k1_2r2 * v_dot_p + k2_2r * r_minus_r0) * dir_dot_p * net, axis=1)  # (num_agents, )

        # 3. Saturation
        u_active = np.clip(u_cs + u_coh, -u_max, u_max)  # (num_agents, )

        # 4. Padding
        u = np.zeros(self.num_agents_max, dtype=np.float32)  # (num_agents_max, )
        u[padding_mask] = u_active  # (num_agents_max, )

        return u

    @staticmethod
    def filter_active_agents_data(data, padding_mask):
        """
        Filters out the data of the inactive agents
        :param data: (num_agents_max, num_agents_max, ...)
        :param padding_mask: (num_agents_max)
        :return: active_data: (num_agents, num_agents, ...)
        """
        # Step 1: Find indices of active agents
        active_agents_indices = np.nonzero(padding_mask)[0]  # (num_agents, )

        # Step 2: Use these indices to index into the data array
        active_data = data[np.ix_(active_agents_indices, active_agents_indices)]

        return active_data

    def update_agent_states(self, state, control_inputs):
        padding_mask = state["padding_mask"]

        # 0. <- 3. Positions
        next_agent_positions = state["agent_states"][:, :2] + state["agent_states"][:, 2:4] * self.dt  # (n_a_max, 2)
        # 1. Headings
        next_agent_headings = state["agent_states"][:, 4] + control_inputs * self.dt  # (num_agents_max, )
        # next_agent_headings = np.mod(next_agent_headings, 2 * np.pi)  # (num_agents_max, )
        # 2. Velocities
        v = self.control_config["speed"]
        next_agent_velocities = np.zeros((self.num_agents_max, 2), dtype=np.float32)  # (num_agents_max, 2)
        next_agent_velocities[padding_mask] = v * np.stack([np.cos(next_agent_headings[padding_mask]),
                                                           np.sin(next_agent_headings[padding_mask])], axis=1)
        # 3. Positions
        # next_agent_positions = state["agent_states"][:, :2] + next_agent_velocities * self.dt  # (num_agents_max, 2)
        # 4. Concatenate
        next_agent_states = np.concatenate(  # (num_agents_max, 5)
            [next_agent_positions, next_agent_velocities, next_agent_headings[:, np.newaxis]], axis=1)

        return next_agent_states  # This did not update the neighbor_masks; it is done in the env_transition

    def compute_neighbor_agents(self, agent_states, padding_mask, communication_range, includes_self_loops=True):
        """
        1. Computes the neighbor matrix based on communication range
        2. Excludes the padding agents (i.e. mask_value==0)
        3. (By default) Includes the self-loops
        """
        self_loop = includes_self_loops  # True if includes self-loops; False otherwise
        agent_positions = agent_states[:, :2]  # (num_agents_max, 2)
        # Get active relative distances
        _, rel_dist = self.get_relative_info(  # (num_agents, num_agents)
            data=agent_positions, mask=padding_mask, get_dist=True, get_active_only=True)
        # Get active neighbor masks
        active_neighbor_masks = rel_dist <= communication_range  # (num_agents, num_agents)
        if not includes_self_loops:
            np.fill_diagonal(active_neighbor_masks, self_loop)  # Set the diagonal to 0 if the mask don't include loops
        # Get the next neighbor masks
        next_neighbor_masks = np.zeros((self.num_agents_max, self.num_agents_max), dtype=np.bool_)  # (num_agents_max, num_agents_max)
        active_agents_indices = np.nonzero(padding_mask)[0]  # (num_agents, )
        next_neighbor_masks[np.ix_(active_agents_indices, active_agents_indices)] = active_neighbor_masks

        # Check no neighbor agents (be careful neighbor mask may not include self-loops)
        neighbor_sum = next_neighbor_masks.sum(axis=1)  # (num_agents_max, )
        if includes_self_loops:
            neighbor_sum -= 1  # Subtract 1 for self-loop
        comm_loss_agents = np.logical_and(padding_mask, neighbor_sum == 0)

        return next_neighbor_masks, comm_loss_agents  # (num_agents_max, num_agents_max), (num_agents_max)

    def get_relative_info(self, data, mask, get_dist=False, get_active_only=False):
        """
        Returns the *relative information(s)* of the agents (e.g. relative position, relative angle, etc.)
        :param data: (num_agents_max, data_dim) ## EVEN IF YOU HAVE 1-D data (i.e. data_dim==1), USE 2-D ARRAY
        :param mask: (num_agents_max)
        :param get_dist:
        :param get_active_only:
        :return: rel_data, rel_dist

        Note:
            - Assumes fully connected communication network
            - If local network needed,
            - Be careful with the **SHAPE** of the input **MASK**;
            - Also, the **MASK** only accounts for the *ACTIVE* agents (similar to padding_mask)
        """

        # Get dimension of the data
        assert data.ndim == 2  # we use a 2D array for the data
        assert data[mask].shape[0] == self.num_agents  # validate the mask
        assert data.shape[0] == self.num_agents_max
        data_dim = data.shape[1]

        # Compute relative data
        # rel_data: shape (num_agents_max, num_agents_max, data_dim); rel_data[i, j] = data[j] - data[i]
        # rel_data_active: shape (num_agents, num_agents, data_dim)
        # rel_data_active := data[mask] - data[mask, np.newaxis, :]
        rel_data_active = data[np.newaxis, mask, :] - data[mask, np.newaxis, :]
        if get_active_only:
            rel_data = rel_data_active
        else:
            rel_data = np.zeros((self.num_agents_max, self.num_agents_max, data_dim), dtype=np.float32)
            rel_data[np.ix_(mask, mask, np.arange(data_dim))] = rel_data_active
            # rel_data[mask, :, :][:, mask, :] = rel_data_active  # not sure; maybe 2-D array (not 3-D) if num_true = 1

        # Compute relative distances
        # rel_dist: shape (num_agents_max, num_agents_max)
        # Note: data are all non-negative!!
        if get_dist:
            rel_dist = np.linalg.norm(rel_data, axis=2) if data_dim > 1 else rel_data.squeeze()
        else:
            rel_dist = None

        # get_active_only==False: (num_agents_max, num_agents_max, data_dim), (num_agents_max, num_agents_max)
        # get_active_only==True: (num_agents, num_agents, data_dim), (num_agents, num_agents)
        # get_dist==False: (n, n, d), None
        return rel_data, rel_dist

    def _compute_rewards(self, state, action, next_state, control_inputs: np.ndarray):
        """
        Compute the rewards; Be careful with the **dimension** of *rewards*
        :param control_inputs: (num_agents_max)
        :return: rewards: (num_agents_max)
        """
        rho = self.control_config["cost_weight"]

        # Heading rate control cost
        heading_rate_costs = (self.dt * self.control_config["speed"]) * np.abs(control_inputs)  # (num_agents_max, )
        # Cruising cost (time penalty)
        cruising_costs = self.dt * np.ones(self.num_agents_max, dtype=np.float32)  # (num_agents_max, )

        rewards = - (heading_rate_costs + (rho * cruising_costs))  # (num_agents_max, )
        rewards[~state["padding_mask"]] = 0

        return rewards  # (num_agents_max, )

    def get_obs(self, state, rel_state, control_inputs):
        """
        Get the observation
        i-th agent's observation: [x, y, vx, vy] with its neighbors' info (and padding info)
        :return: obs
        """
        # This implements it based on Centralized Observations !!!

        # Get masks
        # # We assume that the neighbor_masks are up-to-date and include the paddings (0) and self-loops (1)
        neighbor_masks = state["neighbor_masks"]  # (num_agents_max, num_agents_max); self not included
        padding_mask = state["padding_mask"]
        active_agents_indices = np.nonzero(padding_mask)[0]  # (num_agents, )
        # active_agents_indices_2d = np.ix_(active_agents_indices, active_agents_indices)
        # # Add self-loops only for the active agents
        # neighbor_masks_with_self_loops = neighbor_masks.copy()
        # neighbor_masks_with_self_loops[active_agents_indices_2d] = 1

        # Get p and th  (active agents only)
        active_agent_positions = state["agent_states"][:, :2][active_agents_indices]  # (num_agents, 2)
        # velocities = state["agent_states"][:, 2:4]
        active_agent_headings = state["agent_states"][:, 4][active_agents_indices, np.newaxis]  # (num_agents, 1)
        active_agent_headings = self.wrap_to_pi(active_agent_headings)  # MUST be wrapped to [-pi, pi] before averaging

        # Transform the pos and th into ones in translation-rotation-invariant space
        # # Prepare for the translation and rotation to the swarm coordinate
        center = np.mean(active_agent_positions, axis=0)  # (2, )
        average_heading = np.mean(active_agent_headings)  # (,  ) scalar
        rot_mat = np.array([[np.cos(average_heading), -np.sin(average_heading)],
                            [np.sin(average_heading), np.cos(average_heading)]])  # (2, 2)
        # # Transform p and th
        active_p_transformed = active_agent_positions - center  # (num_agents, 2)  (making the avg pos the origin)
        active_p_transformed = np.dot(active_p_transformed, rot_mat)  # (num_agents, 2)
        # active_p_transformed = np.matmul(active_p_transformed, rot_mat)
        active_th_transformed = active_agent_headings - average_heading  # (num_agents, 1)  (making the avg heading 0)
        # active_th_transformed = self.wrap_to_pi(active_th_transformed)  # (num_agents, 1)

        # # Concat p, |vx|, |vy|, control_inputs; and fill the padding with zeros
        # # # Concat p, |vx|, |vy|, control_inputs
        active_pvu = np.concatenate(
            [active_p_transformed,                        # (num_agents, 2)
             np.cos(active_th_transformed),                       # (num_agents, 1)
             np.sin(active_th_transformed),                       # (num_agents, 1)
             # control_inputs[active_agents_indices, np.newaxis]  # (num_agents, 1) # TODO: what are you going to include in the obs??
             ],
            axis=1
        )  # (num_agents, obs_dim)
        # # # Fill the padding with zeros
        agent_observations = np.zeros((self.num_agents_max, self.obs_dim), dtype=np.float32)
        agent_observations[padding_mask] = active_pvu  # (num_agents_max, obs_dim)

        # Normalize the agent_observations by [ init_bound/2, init_bound/2, 1, 1, u_max ]
        init_bound = self.control_config["initial_position_bound"]
        u_max = self.control_config["max_turn_rate"]
        agent_observations[:, :2] /= (init_bound / 2)
        # agent_observations[:, 4] /= u_max

        # Construct observation
        post_processed_obs = self.post_process_obs(agent_observations, neighbor_masks, padding_mask)
        # # In case of post-processing applied
        if post_processed_obs is not NotImplemented:
            return post_processed_obs
        # # In case of the base implementation (with no post-processing)
        if self.env_mode == "single_env":
            obs = {"centralized_agents_info": agent_observations,  # (num_agents_max, obs_dim)
                   "neighbor_masks": neighbor_masks,              # (num_agents_max, num_agents_max)
                   "padding_mask": padding_mask,                  # (num_agents_max)
                   }
            return obs
        elif self.env_mode == "multi_env":
            multi_obs = {}
            for i in range(self.num_agents_max):
                multi_obs[self.agent_name_prefix + str(i)] = {
                    "centralized_agent_info": agent_observations[i],  # (obs_dim, )
                    "neighbor_mask": neighbor_masks[i],  # (num_agents_max, )
                    "padding_mask": padding_mask,      # (num_agents_max, )
                }
            return multi_obs
        else:
            raise ValueError(f"self.env_mode: 'single_env' or 'multi_env'; not {self.env_mode}; this is get_obs()")

    def post_process_obs(self, agent_observations, neighbor_masks, padding_mask):
        """
        Implement your logic; e.g. flatten the obs for MLP if the MLP doesn't use action masks or so...
        """
        return NotImplemented

    def check_episode_termination(self, state, rel_state, comm_loss_agents):
        """
        Check if the episode is terminated:
            (1) *position std*: sqrt(V(x)+V(y)) < std_p_converged
            (2) *velocity std*: sqrt(V(vx)+V(vy)) < std_v_converged
            (3) *pos std rate*: max of (sqrt(V(x)+V(y))) - min of (sqrt(V(x)+V(y))) < std_p_rate_converged
                in the last 50 iterations
            (4) *vel std rate*: max of (sqrt(V(vx)+V(vy))) - min of (sqrt(V(vx)+V(vy))) < std_v_rate_converged
                in the last 50 iterations
            (5) *max_time_step*: time_step > max_time_step
            (6) *communication*: any agent loses communication with all other agents -> done==True
        :return: done
        """
        padding_mask = state["padding_mask"]
        done = False

        # Compute the current position and velocity std
        agent_positions = state["agent_states"][:, :2][padding_mask]  # (num_agents, 2)
        agent_velocities = state["agent_states"][:, 2:4][padding_mask]  # (num_agents, 2)
        pos_std = np.sqrt(np.sum(np.var(agent_positions, axis=0)))  # scalar
        vel_std = np.sqrt(np.sum(np.var(agent_velocities, axis=0)))  # scalar
        self.std_pos_hist[self.time_step] = pos_std
        self.std_vel_hist[self.time_step] = vel_std

        # Check (1) and (2)
        is_pos_std_converged = pos_std < self.std_p_goal  # (1)
        is_vel_std_converged = vel_std < self.std_v_goal  # (2)
        are_stds_converged = is_pos_std_converged and is_vel_std_converged

        if are_stds_converged and not self.use_fixed_episode_length:
            if self.time_step >= 49:  # magic number!!! (sigh)
                # Check (3)
                last_n_pos_stds = self.std_pos_hist[self.time_step - 49:self.time_step + 1]
                max_pos_std = np.max(last_n_pos_stds)
                min_pos_std = np.min(last_n_pos_stds)
                if max_pos_std - min_pos_std < self.std_p_rate_goal:
                    # Check (4)
                    last_n_vel_stds = self.std_vel_hist[self.time_step - 49:self.time_step + 1]
                    max_vel_std = np.max(last_n_vel_stds)
                    min_vel_std = np.min(last_n_vel_stds)
                    done = max_vel_std - min_vel_std < self.std_v_rate_goal
        # Check (5)
        if self.time_step >= self.max_time_steps-1:
            done = True

        # Check (6)
        if self.comm_range is not None:
            if comm_loss_agents.any():
                done = True
                self.has_lost_comm = True

        if self.env_mode == "single_env":
            return done
        elif self.env_mode == "multi_env":
            # padding agents: False
            dones_in_array = np.ones(self.num_agents_max, dtype=np.bool_)
            # done for swarm agents
            dones_in_array[padding_mask] = done
            dones = self.single_to_multi(dones_in_array)
            # Add "__all__" key to the dones dict
            dones["__all__"] = done
            return dones

    def get_extra_info(self, info, state, rel_state, control_inputs, rewards, done):
        return info

    def compute_custom_reward(self, state, rel_state, control_inputs, rewards, done):
        """
        Impelment your custom reward logic
        :return: custom_reward
        """
        return NotImplemented

    @staticmethod
    def wrap_to_pi(angles):
        """
        Wraps *angles* to **[-pi, pi]**
        """
        return (angles + np.pi) % (2 * np.pi) - np.pi

    def render(self, mode='human'):
        """
        Render the environment
        :param mode:
        :return:
        """
        pass


class LazyMsgListenersTrainEnv(LazyMsgListenersEnv):
    """
    Train environment for the lazy message listeners
    Pendulum-v0 inspired reward function used
    - i.e. (spatial entropy) + (velocity )entropy + (control cost)  # (in a weighted and normalized manner)
    """
    def __init__(self, config):
        super().__init__(config=config)

        # Get pendulum reward weight from config
        # Note: the defualt value is not optimal; they were determined for backward compatibility
        self.w_control = self.config["w_control"] if "w_control" in self.config else 0.02  # 0.001
        self.w_pos = self.config["w_pos"] if "w_pos" in self.config else 1.0  # 1.0
        self.w_vel = self.config["w_vel"] if "w_vel" in self.config else 0.2  # 0.2

    def compute_custom_reward(self, state, rel_state, control_inputs, rewards, done):
        """
        Based on the spatial and velocity entropies with control cost
        :return: custom_reward
        """

        # Get spatial entropy errors
        std_pos = self.std_pos_hist[self.time_step]  # scalar
        std_pos_target = self.std_p_goal - 2.5
        std_pos_error = (std_pos - std_pos_target)**2  # (100-40)**2 = 3600
        pos_error_reward = - (1/3600) * np.maximum(std_pos_error, 0)

        # Get velocity entropy errors
        std_vel = self.std_vel_hist[self.time_step]
        std_vel_target = self.std_v_goal - 0.05
        std_vel_error = (std_vel - std_vel_target)**2  # (15-0.05)**2 = 223.5052
        vel_error_reward = - (1/220) * np.maximum(std_vel_error, 0.0)

        # Get control cost
        rho = self.control_config["cost_weight"]
        control_cost = rewards.sum() / self.num_agents
        control_cost = control_cost + rho * self.dt

        # Get the custom reward
        custom_reward = self.w_pos * pos_error_reward + self.w_vel * vel_error_reward - self.w_control * control_cost

        return custom_reward

    def get_obs(self, state, rel_state, control_inputs):
        obs_org = super().get_obs(state, rel_state, control_inputs)

        # Exclude the last element (control_inputs) from the centralized_agents_info
        agent_observations = obs_org["centralized_agents_info"]  # (num_agents_max, 4)
        obs = {"centralized_agents_info": agent_observations,  # (num_agents_max, 4)
               "neighbor_masks": obs_org["neighbor_masks"],  # (num_agents_max, num_agents_max)
               "padding_mask": obs_org["padding_mask"],      # (num_agents_max)
               }
        return obs


class ProposedFlocking(LazyMsgListenersEnv):

    def __init__(self, config):
        super().__init__(config=config)

    def compute_neighbor_agents(self, agent_states, padding_mask, communication_range, includes_self_loops=False):
        agent_positions = agent_states[:, :2]  # (num_agents_max, 2)
        # Get active relative distances
        _, rel_dist = self.get_relative_info(  # (num_agents, num_agents)
            data=agent_positions, mask=padding_mask, get_dist=True, get_active_only=True)

        # Get active neighbor masks
        num_connections = rel_dist.shape[0]//2

        # Ensure diagonal elements are high to avoid self-connections
        np.fill_diagonal(rel_dist, np.inf)
        # Initialize the boolean array with False
        active_neighbor_masks = np.zeros_like(rel_dist, dtype=bool)
        # Iterate over each row to find the top `num_connections` smallest values
        for i in range(rel_dist.shape[0]):
            # Arg-sort the row to get indices of sorted elements
            sorted_indices = np.argsort(rel_dist[i, :])

            # Mark the top `num_connections` as True
            active_neighbor_masks[i, sorted_indices[:num_connections]] = True

        # For clarity in visualization, set diagonal back to 0
        np.fill_diagonal(rel_dist, 0)

        np.fill_diagonal(active_neighbor_masks, 0)  # REMOVE self-loops; by the definition of neighbor_masks
        # Get the next neighbor masks
        next_neighbor_masks = np.zeros((self.num_agents_max, self.num_agents_max), dtype=np.bool_)  # (num_agents_max, num_agents_max)
        active_agents_indices = np.nonzero(padding_mask)[0]  # (num_agents, )
        next_neighbor_masks[np.ix_(active_agents_indices, active_agents_indices)] = active_neighbor_masks

        # Get no neighbor agents (be careful neighbor mask does not include self-loops)
        comm_loss_agents = np.logical_and(padding_mask, ~next_neighbor_masks.sum(axis=1).astype(np.bool_))

        return next_neighbor_masks, comm_loss_agents  # (num_agents_max, num_agents_max), (num_agents_max)