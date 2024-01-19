# This is this
import gym
from gym.spaces import Box, Discrete, Dict
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


class LazyMsgListenersEnv(gym.Env):
    # TODO (1): Check padding in neighbor_masks creation
    # TODO (2): Take comments out if unnecessary, which might be most of them tho...
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

        # Get the config
        self.config = config
        # env_mode: singe agent (ray) env or multi-agent (ray) env
        self.env_mode = self.config["env_mode"] if "env_mode" in self.config else "single_env"
        # action_type: binary_vector or else
        # # binary_vector: default; action is a binary vector(s) of shape (num_agents_max, num_agents_max)
        # # radius: action is a radius of the communication range; shape (1,) (i.e. scalar)
        # # continuous_vector: action is a continuous vector(s) of shape (num_agents_max, num_agents_max)
        self.action_type = self.config["action_type"] if "action_type" in self.config else "binary_vector"
        # num_agents_pool: pool of possible num_agents;
        # # type: tuple for range
        # # type: ndarray or python list of integers for a list pool of num_agents
        # # note: self.num_agents_pool is transformed to an ndarray in _validate_config()
        # # NO DEFAULT HERE; MUST BE SPECIFIED
        self.num_agents_pool = self.config["num_agents_pool"]
        # agent_name_suffix: suffix of the agent name
        self.agent_name_prefix = self.config["agent_name_prefix"] if "agent_name_prefix" in self.config else "agent_"
        # max_time_step: maximum time step
        self.max_time_step = self.config["max_time_step"] if "max_time_step" in self.config else 1000
        # control_config: config for the C-S flocking control algorithm
        if "control_config" in self.config:
            self.control_config = self.get_default_control_config(self.config["control_config"])
        else:
            self.get_default_control_config()
        self.dt = self.control_config["dt"] if "dt" in self.control_config else 0.1  # time step size
        # comm_range: communication range in meters
        self.comm_range = self.control_config["comm_range"] if "comm_range" in self.control_config else None

        # Define action space

        # Define observation space

        # Other attributes
        # # Number of agents
        self.num_agents = None      # defined in reset()
        self.num_agents_min = None  # defined in _validate_config()
        self.num_agents_max = None  # defined in _validate_config()

        # # States
        # # # state: dict := {"agent_states":   ndarray,  # shape (num_agents_max, data_dim); absolute states!!
        #                     "neighbor_masks": ndarray,  # shape (num_agents_max, num_agents_max)
        #                     "padding_mask":   ndarray,  # shape (num_agents_max)
        # # # # agent_states: [x, y, vx, vy, theta, (u)]; absolute states!!
        # # # # neighbor_masks: 1 if neighbor, 0 if not;  self loop is 0
        # # # # padding_mask:   1 if agent,    0 if padding
        self.state = None
        # # # rel_state: dict := {"rel_agent_positions": ndarray,  # shape (num_agents_max, num_agents_max, 2)
        #                         "rel_agent_velocities": ndarray,  # shape (num_agents_max, num_agents_max, 2)
        #                         "rel_agent_headings": ndarray,  # shape (num_agents_max, num_agents_max)  # 2-D !!!
        #                         "rel_agent_dists": ndarray  # shape (num_agents_max, num_agents_max)
        #                         }
        #  }
        self.rel_state = None
        # # Time steps
        self.time_step = None
        self.agent_time_step = None

        # # Validate the config
        self._validate_config()

    def _validate_config(self):
        # env_mode: must be either "single_env" or "multi_env"
        assert self.env_mode in ["single_env", "multi_env"], "env_mode must be either single_env or multi_env"

        # num_agents_pool: must be a tuple(range)/ndarray of int-s (list is also okay instead of ndarray for list-pool)
        assert isinstance(self.num_agents_pool, (tuple, np.ndarray, list)), "num_agents_pool must be a tuple or ndarray"
        assert all(isinstance(x, int) for x in self.num_agents_pool), "all values in num_agents_pool must be int-s"
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
        # Note: Now self.num_agents_pool is a np-array of possible num_agents; ㅇㅋ?

        # Set num_agents_min and num_agents_max
        self.num_agents_min = self.num_agents_pool.min()
        self.num_agents_max = self.num_agents_pool.max()

        # max_time_step: must be an int and > 0
        assert isinstance(self.max_time_step, int), "max_time_step must be an int"
        assert self.max_time_step > 0, "max_time_step must be > 0"

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
            "communication_decay_rate": 1 / 3,  # Communication decay rate. Default is 1/3
            "cost_weight": 1,  # Cost weight. Default is 1
            "inter_agent_strength": 5,  # Inter agent strength. Default is 5
            "bonding_strength": 1,  # Bonding strength. Default is 1
            "k1": 1,  # K1 coefficient. Default is 1
            "k2": 3,  # K2 coefficient. Default is 3
            "max_turn_rate": 8 / 15,  # Maximum turn rate in rad/s. Default is 8/15
            "initial_position_bound": 250,  # Initial position bound in meters. Default is 250
        }
        if control_config is not None:
            # Check if the control_config has all the keys
            assert all(key in default_control_config for key in control_config.keys()), \
                "control_config must have all the keys of default_control_config"
            # Update the default_control_config with the control_config
            default_control_config.update(control_config)

        return default_control_config

    def reset(self):
        """
        Reset the environment
        :return: obs
        """
        # Init time steps
        self.time_step = 0
        self.agent_time_step = np.zeros(self.num_agents, dtype=np.int32)

        # Get initial num_agents

        # Init the state

        # Get relative state

        pass

    def step(self, action):
        """
        Step the environment
        :param action: your_model_output; ndarray of shape (num_agents_max, num_agents_max) expected under the default
        :return: obs, reward, done, info
        """
        self.time_step += 1
        self.agent_time_step += self.num_agents
        # Get the state of the class (flock)
        state = self.state  # TODO: If deepcopy needed, do it here as shallow copying dict sucks
        # Get rel info
        # # Be careful: rel_state didn't consider the communication network, deliberately
        rel_state = self.rel_state  # TODO: deepcopy it if needed

        # Interpret the action (i.e. model output)
        action_interpreted = self.interpret_action(model_output=action)  # TODO: consider radius action type, *L8R*

        # Multi-agent to single-agent if env_mode == "multi_env"
        joint_action = self.multi_to_single(action_interpreted) if self.env_mode == "multi_env" else action_interpreted
        self.validate_action(action=joint_action,
                             neighbor_mask=state["neighbor_masks"],
                             padding_mask=state["padding_mask"]
                             )

        # Step the environment in *single agent* setting!, which may be faster due to vectorization-like things
        # # s` = T(s, a)
        next_state, control_inputs = self.env_transition(state=state, rel_state=rel_state, action=joint_action)
        next_rel_state = self.get_relative_state(state=next_state)
        # # r = R(s, a, s`)
        rewards = self.compute_rewards(
            state=state, action=joint_action, next_state=next_state, control_inputs=control_inputs)
        reward = rewards.sum() if self.env_mode == "single_env" else self.single_to_multi(rewards)
        # # o = H(s`)
        obs = self.get_obs(state=next_state, rel_state=next_rel_state)
        obs = obs if self.env_mode == "single_env" else self.single_to_multi(obs)

        # Update the method outputs
        # # done
        if self.env_mode == "single_env":
            done = self.check_episode_termination()
        else:
            done = self.check_episode_termination_multi()  # dones; dict
        # # info
        info = self.get_info()

        # Update self.state and the self.rel_state
        self.state = next_state
        self.rel_state = next_rel_state
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
        # rel_agent_velocities, _ = self.get_relative_info(
        #     data=agent_velocities, mask=padding_mask, get_dist=False, get_active_only=False)
        # TODO: replace the below lines with the above line once the class works
        rel_agent_velocities, speeds = self.get_relative_info(
            data=agent_velocities, mask=padding_mask, get_dist=True, get_active_only=False)
        assert np.allclose(speeds, self.control_config["speed"]), "speeds must be equal to self.speed"
        # TODO: remove the assert line once it works

        # Get relative headings
        _, rel_agent_headings = self.get_relative_info(
            data=agent_headings, mask=padding_mask, get_dist=False, get_active_only=False)

        # rel_state: dict
        rel_state = \
            {"rel_agent_positions": rel_agent_positions,
             "rel_agent_velocities": rel_agent_velocities,
             "rel_agent_headings": rel_agent_headings,
             "rel_agent_dists": rel_agent_dists
            }

        return rel_state

    def interpret_action(self, model_output):
        """
        Please implement this method as you need. Currently, it just passes the model_output.
        Interprets the model output
        :param model_output:  # TODO: what is the shape and range of the model_output?
        :return: model_output
        """
        # TODO: do we need to check the shape and range of the model_output?
        return model_output

    def validate_action(self, action, neighbor_mask, padding_mask):
        """
        Validates the action by checking the neighbor_mask and padding_mask
        :param action:  (num_agents_max, num_agents_max)
        :param neighbor_mask: (num_agents_max, num_agents_max)
        :param padding_mask: (num_agents_max)
        :return: None
        """
        # Check the dtype and shape of the action
        assert action.dtype == np.bool_, "action must be a boolean ndarray"
        assert action.shape == (self.num_agents_max, self.num_agents_max), \
            "action must be a boolean ndarray of shape (num_agents_max, num_agents_max)"

        # Check action value based on the neighbor_mask and padding_mask TODO: check if this is correct
        # # action[i, j] == 1 must not found if neighbor_mask[i, j] == 0
        # # action[i, j] == 1 must not found if padding_mask[j] == 0
        assert np.all(np.logical_or(neighbor_mask, ~action)), \
            "action[i, j] == 1 must not found if neighbor_mask[i, j] == 0"
        assert np.all(np.logical_or(padding_mask, ~action)), \
            "action[i, j] == 1 must not found if padding_mask[j] == 0"

        # Ensure the diagonal elements are all zeros (no self-loops)
        assert np.all(np.diag(action) == 0), "Diagonal elements must be zero (no self-loops)"

        # Efficiently check for rows with all zeros (excluding self-loops)
        action_no_diagonals = action.copy()
        np.fill_diagonal(action_no_diagonals, 0)
        assert np.all(action_no_diagonals.sum(axis=1) > 0), \
            "Each row in action, except self-loops, must have at least one True value"

    def to_binary_action(self, action_in_another_type):

        # if self.action_type == "radius":

        pass

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
        assert variable_in_single.shape[0] == self.num_agents, "num_agents must == axis 0 of variable_in_single"
        for i in range(self.num_agents):
            variable_in_multi[self.agent_name_prefix + str(i)] = variable_in_single[i]

        return variable_in_multi

    def env_transition(self, state, rel_state, action):
        """
        Transition the environment; all args in single-rl-agent settings
        s` = T(s, a); deterministic
        :param state: dict:
        :param rel_state: dict:
        :param action: ndarray of shape (num_agents_max, num_agents_max)
        :return: next_state: dict; control_inputs: ndarray of shape (num_agents_max, )
        """
        # Validate the laziness_vectors  TODO: Remove this line once it runs, as it was already checked
        self.validate_action(action=action, neighbor_mask=state["neighbor_masks"], padding_mask=state["padding_mask"])

        # 0. Apply lazy message actions: alters the neighbor_masks!
        lazy_listening_msg_masks = np.logical_and(state["neighbor_masks"], action)  # (num_agents_max, num_agents_max)

        # 1. Get control inputs based on the flocking control algorithm with the lazy listener's network
        control_inputs = self.get_control_inputs(state, rel_state, lazy_listening_msg_masks)

        # 2. Update the agent states based on the control inputs
        next_agent_states = self.update_agent_states(state=state, control_inputs=control_inputs)

        # 3. Update network topology (i.e. neighbor_masks) based on the new agent states
        if self.comm_range is None:
            next_neighbor_masks = state["neighbor_masks"]
        else:
            next_neighbor_masks = self.update_neighbor_agents(
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

        return next_state, control_inputs

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
        r = rel_dist[active_agents_indices_2d]  # (num_agents, num_agents)
        v = rel_vel[active_agents_indices_2d]  # (num_agents, num_agents, 2)
        th = rel_ang[active_agents_indices_2d]  # (num_agents, num_agents)
        th_i = abs_ang[padding_mask]  # (num_agents, )
        net = neighbor_masks[active_agents_indices_2d]  # (num_agents, num_agents) no self-loops (i.e. 0 on diag)
        N = net.sum(axis=1)  # (num_agents, )

        assert np.all(N > 0), "N must be > 0 for all agents"  # TODO: remove this upon the test

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

        next_agent_headings = state("agent_states")[:, 4] + control_inputs * self.dt  # (num_agents_max, )
        # next_agent_headings = np.mod(next_agent_headings, 2 * np.pi)  # (num_agents_max, )

        v = self.control_config["speed"]
        next_agent_velocities = np.zeros((self.num_agents_max, 2), dtype=np.float32)  # (num_agents_max, 2)
        next_agent_velocities = v * np.stack([np.cos(next_agent_headings[padding_mask]),
                                              np.sin(next_agent_headings[padding_mask])], axis=1)

        next_agent_positions = state["agent_states"][:, :2] + next_agent_velocities * self.dt  # (num_agents_max, 2)

        next_agent_states = np.zeros((self.num_agents_max, 5), dtype=np.float32)  # (num_agents_max, 5)
        next_agent_states[:, :2] = next_agent_positions
        next_agent_states[:, 2:4] = next_agent_velocities
        next_agent_states[:, 4] = next_agent_headings

        return next_agent_states

    def update_neighbor_agents(self, agent_states, padding_mask, communication_range):
        agent_positions = agent_states[:, :2]  # (num_agents_max, 2)
        # Get active relative distances
        _, rel_dist = self.get_relative_info(  # (num_agents, num_agents)
            data=agent_positions, mask=padding_mask, get_dist=True, get_active_only=True)
        # Get active neighbor masks
        active_neighbor_masks = rel_dist = communication_range  # (num_agents, num_agents)
        # Get the next neighbor masks
        next_neighbor_masks = np.zeros((self.num_agents_max, self.num_agents_max), dtype=np.bool_)  # (num_agents_max, num_agents_max)
        active_agents_indices = np.nonzero(padding_mask)[0]  # (num_agents, )
        next_neighbor_masks[np.ix_(active_agents_indices, active_agents_indices)] = active_neighbor_masks

        return next_neighbor_masks

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
        assert data[mask].shape[0] == self.num_agents  # TODO: do we have to check both dims?
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

    def compute_rewards(self, state, action, next_state,
                        control_inputs: np.ndarray,
                        ):
        """
        Compute the rewards; Be careful with the **dimension** of *rewards*
        :param control_inputs: (num_agents_max)
        :return: rewards: (num_agents_max)
        """
        w = self.control_config["cost_weight"]

        # Heading rate control cost
        heading_rate_costs = (self.dt * self.control_config["speed"]) * np.abs(control_inputs)  # (num_agents_max, )
        # Cruising cost (time penalty)
        cruising_costs = self.dt * np.ones(self.num_agents_max, dtype=np.float32)

        rewards = - (heading_rate_costs + (w * cruising_costs))  # (num_agents_max, )

        return rewards  # (num_agents_max, )

    def get_obs(self, state, rel_state):
        """
        Get the observation
        i-th agent's observation: [x, y, vx, vy, (u)] with its neighbors' info
        :return: obs
        """
        # Get masks
        # # We assume that the neighbor_masks are up-to-date and include the paddings (0) and self-loops (1)
        neighbor_masks = state["neighbor_masks"]  # (num_agents_max, num_agents_max); self not included
        padding_mask = state["padding_mask"]
        active_agents_indices = np.nonzero(padding_mask)[0]  # (num_agents, )
        active_agents_indices_2d = np.ix_(active_agents_indices, active_agents_indices)
        # Add self-loops only for the active agents
        neighbor_masks_with_self_loops = neighbor_masks.copy()
        neighbor_masks_with_self_loops[active_agents_indices_2d] = 1

        # Get p and th
        positions = state["agent_states"][:, :2]
        # velocities = state["agent_states"][:, 2:4]
        headings = state["agent_states"][:, 4]

        # Transform the pos and th into ones in translation-rotation-invariant space

        #

    def check_episode_termination(self):
        """
        Check if the episode is terminated
        :return: done
        """
        pass

    def check_episode_termination_multi(self):
        """
        Check if the episode is terminated in the multi-agent setting
        Don't forget to add "__all__" key to the dones dict !!
        :return: dones
        """
        pass

    def get_info(self):
        """
        Get the info
        :return: info
        """
        return NotImplemented

    def render(self, mode='human'):
        """
        Render the environment
        :param mode:
        :return:
        """
        pass
