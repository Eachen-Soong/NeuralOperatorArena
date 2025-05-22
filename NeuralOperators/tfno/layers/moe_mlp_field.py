# Inspired by MixtralMoE: https://huggingface.co/docs/transformers/en/model_doc/mixtral

import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import copy

# class MLP(nn.Module):
#     """A Multi-Layer Perceptron, with arbitrary number of layers

#     Parameters
#     ----------
#     in_channels : int
#     out_channels : int, default is None
#         if None, same is in_channels
#     hidden_channels : int, default is None
#         if None, same is in_channels
#     n_layers : int, default is 2
#         number of linear layers in the MLP
#     non_linearity : default is F.gelu
#     dropout : float, default is 0
#         if > 0, dropout probability
#     """

#     def __init__(
#         self,
#         in_channels,
#         out_channels=None,
#         hidden_channels=None,
#         n_layers=2,
#         n_dim=2,
#         non_linearity=F.gelu,
#         dropout=0.0,
#         **kwargs,
#     ):
#         super().__init__()
#         self.n_layers = n_layers
#         self.in_channels = in_channels
#         self.out_channels = in_channels if out_channels is None else out_channels
#         self.hidden_channels = (
#             in_channels if hidden_channels is None else hidden_channels
#         )
#         self.non_linearity = non_linearity
#         self.dropout = (
#             nn.ModuleList([nn.Dropout(dropout) for _ in range(n_layers)])
#             if dropout > 0.0
#             else None
#         )

#         Conv = getattr(nn, f"Conv{n_dim}d")
#         self.fcs = nn.ModuleList()
#         for i in range(n_layers):
#             if i == 0 and i == (n_layers - 1):
#                 self.fcs.append(Conv(self.in_channels, self.out_channels, 1))
#             elif i == 0:
#                 self.fcs.append(Conv(self.in_channels, self.hidden_channels, 1))
#             elif i == (n_layers - 1):
#                 self.fcs.append(Conv(self.hidden_channels, self.out_channels, 1))
#             else:
#                 self.fcs.append(Conv(self.hidden_channels, self.hidden_channels, 1))

#     def forward(self, x):
#         for i, fc in enumerate(self.fcs):
#             x = fc(x)
#             if i < self.n_layers - 1:
#                 x = self.non_linearity(x)
#             if self.dropout is not None:
#                 x = self.dropout[i](x)

#         return x

class MLP(nn.Module):
    def __init__(self, ffn_dim, hidden_dim, activation):
        super().__init__()
        self.ffn_dim = ffn_dim
        self.hidden_dim = hidden_dim

        self.w1 = nn.Linear(self.hidden_dim, self.ffn_dim, bias=False)
        self.w2 = nn.Linear(self.ffn_dim, self.hidden_dim, bias=False)
        self.w3 = nn.Linear(self.hidden_dim, self.ffn_dim, bias=False)

        self.act_fn = activation

    def forward(self, hidden_states):
        current_hidden_states = self.act_fn(self.w1(hidden_states)) * self.w3(hidden_states)
        current_hidden_states = self.w2(current_hidden_states)
        return current_hidden_states

class MoeMLP(nn.Module):
    """
    This implementation is
    strictly equivalent to standard MoE with full capacity (no
    dropped tokens). It's faster since it formulates MoE operations
    in terms of block-sparse operations to accomodate imbalanced
    assignments of tokens to experts, whereas standard MoE either
    (1) drop tokens at the cost of reduced performance or (2) set
    capacity factor to number of experts and thus waste computation
    and memory on padding.
    """

    def __init__(self, in_channels,
                    out_channels=None,
                    hidden_channels=None, 
                    n_layers=2,
                    n_dim=2,
                    non_linearity=F.gelu,
                    dropout=0.0,
                    num_experts=8, top_k=2, jitter_noise=0.):
        super().__init__()
        self.n_layers = n_layers
        self.in_channels = in_channels
        self.out_channels = in_channels if out_channels is None else out_channels
        self.hidden_channels = (
            in_channels if hidden_channels is None else hidden_channels
        )
        self.n_dim = n_dim
        self.non_linearity = non_linearity
        self.num_experts = num_experts
        self.top_k = top_k

        # gating
        # Conv = getattr(nn, f"Conv{n_dim}d")
        # self.gate = Conv(self.in_channels, self.num_experts, 1, bias=False)
        self.gate = nn.Linear(self.hidden_channels, self.num_experts, bias=False)
        

        # self.experts = nn.ModuleList([MLP(in_channels=in_channels, out_channels=out_channels, n_layers=n_layers, n_dim=n_dim, non_linearity=non_linearity, dropout=dropout) for _ in range(self.num_experts)])
        self.experts = nn.ModuleList([MLP(ffn_dim=self.in_channels, hidden_dim=self.out_channels, activation=self.non_linearity) for _ in range(self.num_experts)])

        # Jitter parameters
        self.jitter_noise = jitter_noise
        
        self.permuter = [0] + list(range(2, self.n_dim + 2)) + [1]
        self.inverse_premuter = [0, -1] + list(range(1, self.n_dim + 1))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
            hidden_states: (b, c, [space_dims])
        """
        shape = hidden_states.shape
        batch_size = shape[0]; hidden_dim = shape[1]; space_dim = shape[2:]
        permuter = copy(self.permuter)
        hidden_states = hidden_states.permute(*permuter).reshape(-1, hidden_dim)

        if self.training and self.jitter_noise > 0:
            hidden_states *= torch.empty_like(hidden_states).uniform_(1.0 - self.jitter_noise, 1.0 + self.jitter_noise)
        # router_logits: (batch, n_experts, [space_dim])
        router_logits = self.gate(hidden_states)
        

        routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
        routing_weights, selected_experts = torch.topk(routing_weights, self.top_k, dim=1)
        routing_weights /= routing_weights.sum(dim=1, keepdim=True)
        # we cast back to the input dtype
        routing_weights = routing_weights.to(hidden_states.dtype)
        
        spacial_resolution = 1
        for res in space_dim:
            spacial_resolution *= res

        final_hidden_states = torch.zeros(
            (batch_size * spacial_resolution, hidden_dim), dtype=hidden_states.dtype, device=hidden_states.device
        )

        # One hot encode the selected experts to create an expert mask
        # this will be used to easily index which expert is going to be sollicitated

        # permuter = self.permuter; permuter[0] = 1; permuter[1] = 0; permuter.insert(0, -1)
        permuter = permuter + [-1]
        expert_mask = torch.nn.functional.one_hot(selected_experts, num_classes=self.num_experts)
        # expert_mask = expert_mask.permute(*permuter).reshape(-1, self.num_experts, self.top_k).permute(1, 2, 0)
        expert_mask = expert_mask.permute(2, 1, 0)

        # Loop over all available experts in the model and perform the computation on each expert
        for expert_idx in range(self.num_experts):
            expert_layer = self.experts[expert_idx]

            idx, top_x = torch.where(expert_mask[expert_idx])
            current_state = hidden_states[None, top_x].reshape(-1, hidden_dim)
            
            if not len(current_state):
                continue
            current_hidden_states = expert_layer(current_state) * routing_weights[top_x, idx, None]


            final_hidden_states.index_add_(0, top_x, current_hidden_states.to(hidden_states.dtype))
        final_hidden_states = final_hidden_states.reshape(batch_size, *space_dim, hidden_dim).permute(*self.inverse_premuter)
        # return final_hidden_states, router_logits
        return final_hidden_states
