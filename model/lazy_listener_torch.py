import copy
from typing import Dict, List, Union
from ray.rllib.utils.typing import ModelConfigDict, TensorType
import numpy as np

# PyTorch
import torch
import torch.nn as nn

# Custom modules




class MsgLazinessAllocatorPPO(nn.Module):
    def __init__(self, input_embedding_layer, encoder, decoder, generator, model_config):
        super().__init__()
        self.input_embedding_layer = input_embedding_layer
        self.encoder = encoder
        self.decoder = decoder
        self.generator = generator
        self.model_config = model_config

    def forward(self, obs_dict: Dict[str, TensorType]):

        debug = True

        # Get data
        agents_info = obs_dict["centralized_agents_info"]  # (batch_size, num_agents_max, d_subobs)
        pad_mask_tokens = obs_dict["padding_mask"]  # (batch_size, num_agents_max)
        neighbor_masks_tokens = obs_dict["neighbor_masks"]  # (batch_size, num_agents_max, num_agents_max)
        # TODO: Don't forget to think about SELF-LOOPING agents

        # Get masks  # TODO: turn off dim_check once working
        # # encoder mask
        encoder_mask = self.make_mask_from_local_keys(query=pad_mask_tokens, key=pad_mask_tokens,
                                                      network=neighbor_masks_tokens, pad_idx=0, disconnection_idx=0,
                                                      dim_check=debug)
        # # decoder masks
        tgt_mask = None  # subsequent mask not used in this application


        # Encode

        # Get context

        # Decode

        # Generate

        return logits, value_branch_input, info

    def encode(self, src, src_mask):
        return self.encoder(self.input_embedding_layer(src), src_mask)

    def get_context(self, endocer_out, padding_mask):


        return context

    def decode(self, tgt, encoder_out, tgt_mask, src_tgt_mask):
        return self.decoder(tgt, encoder_out, tgt_mask, src_tgt_mask)

    def generate(self, query, key, mask):
        return self.generator(query, key, mask)

    # def make_pad_mask(self, query, key, pad_idx=1, dim_check=False):
    #     # query: (n_batch, query_seq_len)
    #     # key: (n_batch, key_seq_len)
    #     # If input_token==pad_idx, then the mask value is 0, else 1
    #     # In the MHA layer, (no attention) == (attention_score: -inf) == (mask value is 0) == (input_token==pad_idx)
    #     # WARNING: Choose pad_idx carefully, particularly about the data type (e.g. float, int, ...)
    #
    #     # Check if the query and key have the same dimension
    #     if dim_check:
    #         assert len(query.shape) == 2, "query must have 2 dimensions: (n_batch, query_seq_len)"
    #         assert len(key.shape) == 2, "key must have 2 dimensions: (n_batch, key_seq_len)"
    #         assert query.size(0) == key.size(0), "query and key must have the same batch size"
    #
    #     query_seq_len, key_seq_len = query.size(1), key.size(1)
    #
    #     query_mask = query.ne(pad_idx).unsqueeze(1).unsqueeze(3)  # (n_batch, 1, query_seq_len, 1)
    #     query_mask = query_mask.repeat(1, 1, 1, key_seq_len)      # (n_batch, 1, query_seq_len, key_seq_len)
    #
    #     key_mask = key.ne(pad_idx).unsqueeze(1).unsqueeze(2)  # (n_batch, 1, 1, key_seq_len); on the same device as key
    #     key_mask = key_mask.repeat(1, 1, query_seq_len, 1)    # (n_batch, 1, query_seq_len, key_seq_len)
    #
    #     mask = key_mask & query_mask
    #     mask.requires_grad = False
    #     return mask  # output shape: (n_batch, 1, query_seq_len, key_seq_len)  1 for head dimension in MHA

    def make_mask_from_local_keys(self, query, key, network, pad_idx=1, disconnection_idx=0, dim_check=False):
        # query: (n_batch, query_seq_len)
        # key: (n_batch, key_seq_len)
        # network: (n_batch, query_seq_len, key_seq_len)
        # If input_token==pad_idx, then the mask value is 0, else 1
        # In the MHA layer, (no attention) == (attention_score: -inf) == (mask value is 0) == (input_token==pad_idx)
        # WARNING: Choose pad_idx carefully, particularly about the data type (e.g. float, int, ...)

        query_seq_len, key_seq_len = query.size(1), key.size(1)

        # Check if the query and key have the same dimension
        if dim_check:
            assert len(query.shape) == 2, "query must have 2 dimensions: (n_batch, query_seq_len)"
            assert len(key.shape) == 2, "key must have 2 dimensions: (n_batch, key_seq_len)"
            assert query.size(0) == key.size(0), "query and key must have the same batch size"
            assert len(network.shape) == 3, "network must have 3 dimensions: (n_batch, query_seq_len, key_seq_len)"
            assert network.size(0) == query.size(0), "network and query must have the same batch size"
            assert network.size(1) == query_seq_len, "network and query must have the same sequence length"
            assert network.size(2) == key_seq_len, "network and key must have the same sequence length"

        # Make query mask
        query_mask = query.ne(pad_idx).unsqueeze(1).unsqueeze(3)  # (n_batch, 1, query_seq_len, 1); same device as query
        query_mask = query_mask.repeat(1, 1, 1, key_seq_len)      # (n_batch, 1, query_seq_len, key_seq_len)

        # Make key mask; also use network
        key_mask = key.ne(pad_idx).unsqueeze(1).unsqueeze(2)     # (n_batch, 1, 1, key_seq_len); on t same device as key
        key_mask = key_mask.repeat(1, 1, query_seq_len, 1)       # (n_batch, 1, query_seq_len, key_seq_len)
        local_mask = network.ne(disconnection_idx).unsqueeze(1)  # (n_batch, 1, query_seq_len, key_seq_len)
        local_key_mask = key_mask & local_mask                   # (n_batch, 1, query_seq_len, key_seq_len)

        mask = local_key_mask & query_mask
        mask.requires_grad = False
        return mask  # output shape: (n_batch, 1, query_seq_len, key_seq_len)  1 for head dimension in MHA


class MJActorTest(nn.Module):
    def __init__(self, input_dim):
        super().__init__()

        self.embedding_dim = 128
        self.agent_embedding_dim = 128
        self.n_enc_layer = 3
        self.n_head = 8
        self.ff_dim = 512
        self.norm_eps = 1e-5

        self.flock_embedding = nn.Linear(input_dim, self.embedding_dim)

        ## Encoder
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=self.embedding_dim,
                                                        nhead=self.n_head,
                                                        dim_feedforward=self.ff_dim,
                                                        dropout=0.0,
                                                        layer_norm_eps=self.norm_eps,
                                                        norm_first=True,
                                                        batch_first=True)

        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=self.n_enc_layer,
                                             enable_nested_tensor=False)

        self.Wq = nn.Parameter(torch.randn(self.embedding_dim * 2, self.embedding_dim))
        self.Wk = nn.Parameter(torch.randn(self.embedding_dim, self.embedding_dim))

        self.tanh = nn.Tanh()

    def forward(self, obs_dict: Dict[str, TensorType]):
        # Get data
        agents_info = obs_dict["centralized_agents_info"]  # (batch_size, num_agents_max, d_subobs)

        # run MJ's forward
        att_scores = self.mj_forward_actor(agents_info)  # [batch_size, n_agent, n_agent]

        return att_scores

    def mj_forward_actor(self, obs):
        """
        Input:
            obs: [batch_size, n_agent, input_dim]
        """

        batch_size = obs.shape[0]
        n_agent = obs.shape[1]

        flock_embed = self.flock_embedding(obs)  # [batch_size, n_agent, embedding_dim]

        # # encoder1
        enc = self.encoder(flock_embed)  # [batch_size, n_agent, embedding_dim]

        # context embedding
        context = torch.mean(enc, dim=1)  # [batch_size, embedding_dim]

        flock_context = context.unsqueeze(1).expand(batch_size, n_agent,
                                                    context.shape[-1])  # [batch_size, n_agent, embedding_dim]
        agent_context = torch.cat((enc, flock_context), dim=-1)  # [batch_size, n_agent, embedding_dim*2]

        queries = torch.matmul(agent_context, self.Wq)  # [batch_size, n_agent, embedding_dim]
        keys = torch.matmul(enc, self.Wk)  # [batch_size, n_agent, embedding_dim]
        D = queries.shape[-1]

        # attention
        att_scores = torch.bmm(queries, keys.transpose(1, 2)) / np.sqrt(D)  # [batch_size, n_agent, n_agent]
        # att_scores = (self.tanh(att_scores) + 1) / 2
        #
        # # make diagonal elements to 1
        # ones = att_scores.new_ones(att_scores.shape[1])  # [n_agent, n_agent]
        # I_mat = torch.diag_embed(ones).expand_as(att_scores)  # [batch_size, n_agent, n_agent]
        # att_scores = att_scores * (1 - I_mat) + I_mat  # [batch_size, n_agent, n_agent]
        #
        # return att_scores

        #
        # Fill the diagonal with very large positive value (to make the corresponding probability close to 1)
        large_val = 1e9  # may cause NaN if it passes through softmax (or exp)
        # large_val = 512
        att_scores *= 2e-3
        att_scores = att_scores - torch.diag_embed(att_scores.new_full((n_agent,), large_val))  # [batch_size, n_agent, n_agent]

        # large_val = 1e9
        # ones = att_scores.new_ones(att_scores.shape[1])  # [n_agent, n_agent]
        # I_mat = torch.diag_embed(ones).expand_as(att_scores)  # [batch_size, n_agent, n_agent]
        # att_scores = att_scores * (1 - I_mat) + (large_val * I_mat)  # [batch_size, n_agent, n_agent]


        # negate the scores (representing the probability of action being 0 in the softmax)
        neg_att_scores = - att_scores  # [batch_size, n_agent, n_agent]

        z_expanded = att_scores.unsqueeze(-1)  # [batch_size, n_agent, n_agent, 1]
        neg_z_expanded = neg_att_scores.unsqueeze(-1)  # [batch_size, n_agent, n_agent, 1]

        # Concat along the new dim
        z_cat = torch.cat((z_expanded, neg_z_expanded), dim=-1)  # [batch_size, n_agent, n_agent, 2]
        # z_cat = torch.cat((neg_z_expanded, z_expanded), dim=-1)  # [batch_size, n_agent, n_agent, 2]

        # Reshape to 2D (batch_size, 2* n_agent*n_agent
        z_reshaped = z_cat.reshape(batch_size, n_agent * n_agent * 2)  # [batch_size, n_agent*n_agent*2]

        return z_reshaped  # [batch_size, n_agent*n_agent*2]


class MJCriticTest(nn.Module):
    def __init__(self, input_dim):
        super().__init__()

        self.embedding_dim = 128
        self.agent_embedding_dim = 128
        self.n_enc_layer = 3
        self.n_head = 8
        self.ff_dim = 512
        self.norm_eps = 1e-5

        self.flock_embedding = nn.Linear(input_dim, self.embedding_dim)

        ## Encoder
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=self.embedding_dim,
                                                        nhead=self.n_head,
                                                        dim_feedforward=self.ff_dim,
                                                        dropout=0.0,
                                                        layer_norm_eps=self.norm_eps,
                                                        norm_first=True,
                                                        batch_first=True)

        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=self.n_enc_layer,
                                             enable_nested_tensor=False)

        self.value_net = nn.Linear(self.embedding_dim, 1)

    def forward(self, obs_dict: Dict[str, TensorType]):
        # Get data
        agents_info = obs_dict["centralized_agents_info"]  # (batch_size, num_agents_max, d_subobs)

        # run MJ's forward
        value_unsqueezed = self.mj_forward_critic(agents_info)  # [batch_size, 1]

        return value_unsqueezed  # [batch_size, 1]

    def mj_forward_critic(self, obs):
        """
        Input:
            obs: [batchsize, n_agent, input_dim]
        """

        flock_embed = self.flock_embedding(obs)  # [batch_size, n_agent, embedding_dim]

        # # encoder1
        enc = self.encoder(flock_embed)  # [batch_size, n_agent, embedding_dim]

        # context embedding
        context = torch.mean(enc, dim=1)  # [batch_size, embedding_dim]

        value = self.value_net(context)  # [batch_size, 1]

        return value  # [batch_size, 1]

