# Everything is copy
import copy
# Please let me get out of ray rllib
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2, ModelV2
from ray.rllib.utils.typing import ModelConfigDict, TensorType
from ray.rllib.utils.annotations import override
from ray.rllib.models.preprocessors import get_preprocessor
from ray.rllib.models.torch.misc import SlimFC, normc_initializer
#
# From envs
import numpy as np
from gym.spaces import Discrete
from typing import List, Union
#
# Pytorch
import torch
import torch.nn as nn
#
# For the custom model
from model.lazy_listener_torch import MJActorTest, MJCriticTest


class LazyListenerModelTemplate(TorchModelV2, nn.Module):
    """
    Abstract class of models for LazyMsgListenersEnv
    Extend this class to implement your custom model with your requirements (e.g. RL-algos, attention flows, etc.)
    """
    def __init__(self, obs_space, action_space, num_outputs, model_config, name, **kwargs):
        nn.Module.__init__(self)  # Initialize nn.Module first
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)

        # Get model_config from outer scope
        model_config_validated = self._validate_config(model_config)
        self._set_attribute_from_model_config(model_config_validated)  # sets class attributes in place

        # Define layers
        _modules = self._define_modules_from_config(model_config_validated)

        # Define the model
        self.policy_network = None
        self.value_network = None
        self.values = None
        self.value_branch = None
        self.use_shared_value_network = model_config_validated["use_shared_value_network"]
        self._build_model_from_modules(_modules)

        # Check if the model is built correctly
        assert self.policy_network is not None and self.value_network is not None, "Model is not built correctly"

    def _validate_config(self, model_config: ModelConfigDict):
        """
        Configuration Template:
        "custom_model_config":{
            "show_your_custom_config": "here"
        }
        """
        raise NotImplementedError

    def _set_attribute_from_model_config(self, model_config: ModelConfigDict):
        """
        Set attributes from model_config
        Assumes the model_config is validated
        :param model_config:
        :return:
        """
        raise NotImplementedError

    def _define_modules_from_config(self, model_config: ModelConfigDict):
        raise NotImplementedError

    def _build_model_from_modules(self, modules: dict):
        raise NotImplementedError

    def forward(
        self,
        input_dict,
        state: List[TensorType],
        seq_lens: TensorType
    ) -> (TensorType, List[TensorType]):

        # Get and check the observation
        obs = input_dict["obs"]
        self._validate_observation(obs)

        # Forward pass through the model(s)
        if self.use_shared_value_network:  # Shared value network
            x, value_branch_input, info = self.policy_network(obs)
            self.values = value_branch_input  # YOU MUST CHECK THE DIMENSIONALITY; (batch_size, ... )
        else:                              # Separate value network
            x, _, info = self.policy_network(obs)
            self.values = self.value_network(obs)

        # Check number of outputs
        assert x.shape[-1] == self.num_outputs, "Number of outputs is not valid"

        return x, state

    def _validate_observation(self, obs: TensorType):
        raise NotImplementedError

    def value_function(self) -> TensorType:
        raise NotImplementedError


class LazyListenerModelPPO(LazyListenerModelTemplate):
    """
    Custom model for LazyMsgListenersEnv with PPO
    """
    def _validate_config(self, model_config: ModelConfigDict):
        """
        Configuration Template:
        "custom_model_config":{
            # Mandatory key-value pairs
            "use_shared_value_network": False
            "d_subobs": 4,  # No default value; from obs_space['centralized_agents_info'].shape[1]
            # Optional key-value pairs
            "d_embed_input": 128,
            "d_embed_context": 128,
            "d_model": 128,
            "d_model_decoder": 128,
            "n_heads": 8,
            "d_ff": 256,
            "d_ff_decoder": 256,
            "n_layers_encoder": 3,
            "n_layers_decoder": 1,
            "use_bias_in_obs_embedding": True,
            "use_bias_in_w_qkv": False,
            "layer_norm_eps": 1e-5,
            "dropout_rate": 0,
        }
        """
        assert "custom_model_config" in model_config, "custom_model_config is not found in model_config"
        cfg = model_config["custom_model_config"]

        # Check mandatory keys
        # # keys?
        assert "use_shared_value_network" in cfg, "use_shared_value_network is not found in custom_model_config"
        assert "d_subobs" in cfg, "d_subobs is not found in custom_model_config"
        # # types?
        assert isinstance(cfg["use_shared_value_network"], bool), "use_shared_value_network must be a boolean"
        assert isinstance(cfg["d_subobs"], int), "d_subobs must be an integer"
        # # valid values?
        assert cfg["d_subobs"] == self.obs_space["centralized_agents_info"].shape[1], "d_subobs is not valid"

        # Check optional keys
        # # have keys?  if not, set default values
        cfg["d_embed_input"] = cfg.get("d_embed_input", 128)
        cfg["d_embed_context"] = cfg.get("d_embed_context", 128)
        cfg["d_model"] = cfg.get("d_model", 128)
        cfg["d_model_decoder"] = cfg.get("d_model_decoder", 128)
        cfg["n_heads"] = cfg.get("n_heads", 8)
        cfg["d_ff"] = cfg.get("d_ff", 256)
        cfg["d_ff_decoder"] = cfg.get("d_ff_decoder", 256)
        cfg["n_layers_encoder"] = cfg.get("n_layers_encoder", 3)
        cfg["n_layers_decoder"] = cfg.get("n_layers_decoder", 1)
        cfg["use_bias_in_obs_embedding"] = cfg.get("use_bias_in_obs_embedding", True)
        cfg["use_bias_in_w_qkv"] = cfg.get("use_bias_in_w_qkv", False)
        cfg["layer_norm_eps"] = cfg.get("layer_norm_eps", 1e-5)
        cfg["dropout_rate"] = cfg.get("dropout_rate", 0)
        # # types?
        assert isinstance(cfg["d_embed_input"], int), "d_embed_input must be an integer"
        assert isinstance(cfg["d_embed_context"], int), "d_embed_context must be an integer"
        assert isinstance(cfg["d_model"], int), "d_model must be an integer"
        assert isinstance(cfg["d_model_decoder"], int), "d_model_decoder must be an integer"
        assert isinstance(cfg["n_heads"], int), "n_heads must be an integer"
        assert isinstance(cfg["d_ff"], int), "d_ff must be an integer"
        assert isinstance(cfg["d_ff_decoder"], int), "d_ff_decoder must be an integer"
        assert isinstance(cfg["n_layers_encoder"], int), "n_layers_encoder must be an integer"
        assert isinstance(cfg["n_layers_decoder"], int), "n_layers_decoder must be an integer"
        assert isinstance(cfg["use_bias_in_obs_embedding"], bool), "use_bias_in_obs_embedding must be a boolean"
        assert isinstance(cfg["use_bias_in_w_qkv"], bool), "use_bias_in_w_qkv must be a boolean"
        assert isinstance(cfg["layer_norm_eps"], float), "layer_norm_eps must be a float"
        assert isinstance(cfg["dropout_rate"], float), "dropout_rate must be a float"
        # # valid values?
        assert cfg["d_embed_input"] > 0, "d_embed_input must be a positive integer"
        assert cfg["d_embed_context"] > 0, "d_embed_context must be a positive integer"
        assert cfg["d_model"] > 0, "d_model must be a positive integer"
        assert cfg["d_model_decoder"] > 0, "d_model_decoder must be a positive integer"
        assert cfg["n_heads"] > 0, "n_heads must be a positive integer"
        assert cfg["d_ff"] > 0, "d_ff must be a positive integer"
        assert cfg["d_ff_decoder"] > 0, "d_ff_decoder must be a positive integer"
        assert cfg["n_layers_encoder"] > 0, "n_layers_encoder must be a positive integer"
        assert cfg["n_layers_decoder"] > 0, "n_layers_decoder must be a positive integer"
        assert cfg["layer_norm_eps"] > 0, "layer_norm_eps must be a positive float"
        assert 0 <= cfg["dropout_rate"] < 1, "dropout_rate must be in [0, 1)"

        return cfg

    def _set_attribute_from_model_config(self, model_config: ModelConfigDict):
        pass

    def _define_modules_from_config(self, model_config: ModelConfigDict):  # used in __init__ of the parent class
        # # Define the policy network
        # policy_network =
        # # Define the value network
        # value_network =
        # # Define the value branch
        # value_branch =
        return {
            # "policy_network": policy_network,
            # "value_network": value_network,
            # "value_branch": value_branch
        }

    def _build_model_from_modules(self, modules: dict):  # used in __init__ of the parent class
        self.policy_network = modules["policy_network"]
        self.value_network = modules["value_network"]
        self.value_branch = modules["value_branch"]

    def _validate_observation(self, obs: TensorType):
        pass  # TODO: Implement this

    def value_function(self) -> TensorType:
        value = self.value_branch(self.values).squeeze(-1)  # (batch_size,)
        assert value.shape == (self.values.shape[0],), "value is not valid"  # TODO: Remove this line
        return value


class LazyListenerModelPPOTestMJ(TorchModelV2, nn.Module):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name, **kwargs):
        nn.Module.__init__(self)  # Initialize nn.Module first
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)

        # Get actor
        # assert obs_space["centralized_agents_info"].shape[2] == 4, "d_subobs is not 4"
        # self.actor = MJActorTest(obs_space["centralized_agents_info"].shape[2])
        self.actor = MJActorTest(4)

        # Get critic
        # self.critic = MJCriticTest(obs_space["centralized_agents_info"].shape[2])
        self.critic = MJCriticTest(4)
        self.values = None

    def forward(
            self,
            input_dict,
            state: List[TensorType],
            seq_lens: TensorType
    ) -> (TensorType, List[TensorType]):
        # Get and check the observation
        obs_dict = input_dict["obs"]

        x = self.actor(obs_dict)  # (batch_size, num_agents_max * num_agents_max * 2)
        self.values = self.critic(obs_dict)  # (batch_size, 1)

        return x, state

    def value_function(self) -> TensorType:
        value = self.values.squeeze(-1)  # (batch_size,)
        return value

