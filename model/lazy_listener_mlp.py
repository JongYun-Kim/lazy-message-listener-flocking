import torch
import torch.nn as nn
from typing import List

from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.utils.annotations import override
from ray.rllib.models.torch.misc import SlimFC, normc_initializer
from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.utils.typing import TensorType


class InfoLazyMLP(TorchModelV2, nn.Module):
    """
    Custom model for MLP ablation study
    Input size: (batch_size, num_agents_max * d_subobs) == (B, 20 * 4) == (B, 80)
    FC policy pre-output: (batch_size, 400) == (B, 20 * 20)
    Final logits: (batch_size, 800) w/ neg trick
    Value output: (batch_size,) scalar
    """
    def __init__(self, obs_space, action_space, num_outputs, model_config, name, **kwargs):
        nn.Module.__init__(self)
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)

        # constants
        self.n_agent = obs_space.original_space["centralized_agents_info"].shape[0]
        self.d_subobs = obs_space.original_space["centralized_agents_info"].shape[1]
        # sanity: expect n_agent=20, d_subobs=4
        flat_input_size = self.n_agent * self.d_subobs  # 80

        # Policy network: 80 -> 256 -> 512 -> 1024 -> 400
        self.policy_fc1 = SlimFC(
            in_size=flat_input_size,
            out_size=256,
            activation_fn=torch.nn.ReLU,
            initializer=normc_initializer(1.0),
            use_bias=True,
        )
        self.policy_fc2 = SlimFC(
            in_size=256,
            out_size=512,
            activation_fn=torch.nn.ReLU,
            initializer=normc_initializer(1.0),
            use_bias=True,
        )
        self.policy_fc3 = SlimFC(
            in_size=512,
            out_size=1024,
            activation_fn=torch.nn.ReLU,
            initializer=normc_initializer(1.0),
            use_bias=True,
        )
        # pre-logit policy output (to be reshaped into att_scores)
        self.policy_fc4 = SlimFC(
            in_size=1024,
            out_size=self.n_agent * self.n_agent,  # 400
            activation_fn=None,
            initializer=normc_initializer(0.01),
            use_bias=True,
        )

        # Value network: 80 -> 128 -> 128 -> 64 -> 1
        self.value_fc1 = SlimFC(
            in_size=flat_input_size,
            out_size=128,
            activation_fn=torch.nn.ReLU,
            initializer=normc_initializer(1.0),
            use_bias=True,
        )
        self.value_fc2 = SlimFC(
            in_size=128,
            out_size=128,
            activation_fn=torch.nn.ReLU,
            initializer=normc_initializer(1.0),
            use_bias=True,
        )
        self.value_fc3 = SlimFC(
            in_size=128,
            out_size=64,
            activation_fn=torch.nn.ReLU,
            initializer=normc_initializer(1.0),
            use_bias=True,
        )
        self.value_fc4 = SlimFC(
            in_size=64,
            out_size=1,
            activation_fn=None,
            initializer=normc_initializer(0.01),
            use_bias=True,
        )
        # placeholder for value
        self._value = None

    @override(ModelV2)
    def forward(self, input_dict, state: List[TensorType], seq_lens: TensorType):
        obs = input_dict["obs"]["centralized_agents_info"]  # [B, n_agent, d_subobs]
        batch_size = obs.shape[0]
        x = obs.view(batch_size, -1)  # Flatten to [B, 80]

        # --- Policy path ---
        p = self.policy_fc1(x)
        p = self.policy_fc2(p)
        p = self.policy_fc3(p)
        mlp_scores = self.policy_fc4(p)  # [B, n_agent * n_agent]
        # reshape to [B, n_agent, n_agent]
        mlp_scores = mlp_scores.view(batch_size, self.n_agent, self.n_agent)

        # apply scaling and diag trick
        policy_score_scale = 2e-3
        mlp_scores = mlp_scores * policy_score_scale

        large_val = 1e9  # to strongly bias diagonal
        # subtract huge value on diagonal to push those probabilities (after softmax) toward 1 for the positive branch
        mlp_scores = mlp_scores - torch.diag_embed(
            mlp_scores.new_full((self.n_agent,), large_val)
        )  # [B, n_agent, n_agent]

        neg_mlp_scores = -mlp_scores  # [B, n_agent, n_agent]

        z_expanded = mlp_scores.unsqueeze(-1)       # [B, n_agent, n_agent, 1]
        neg_z_expanded = neg_mlp_scores.unsqueeze(-1)  # [B, n_agent, n_agent, 1]
        z_cat = torch.cat((z_expanded, neg_z_expanded), dim=-1)  # [B, n_agent, n_agent, 2]
        logits = z_cat.reshape(batch_size, self.n_agent * self.n_agent * 2)  # [B, 800]

        # --- Value path ---
        v = self.value_fc1(x)
        v = self.value_fc2(v)
        v = self.value_fc3(v)
        v = self.value_fc4(v)  # [B, 1]
        # store value as flat tensor [B]
        self._value = v

        return logits, state

    @override(ModelV2)
    def value_function(self) -> TensorType:
        if self._value is None:
            # RLlib usually calls forward before this, so this is a safety.
            raise ValueError(
                "value_function called before forward, which should not happen. "
                "Please check your model implementation."
            )
        return self._value.squeeze(-1)  # [B]

