# Adapted from MIST-CF (Goldman et al., 2023)
# Licensed under MIT License - see LICENSE in this directory
"""nn_utils.py"""

import copy
import math

import torch
import torch.nn as nn


def build_lr_scheduler(optimizer, lr_decay_rate, decay_steps=5000, warmup=100):
    def lr_lambda(step):
        if step >= warmup:
            step = step - warmup
            rate = lr_decay_rate ** (step // decay_steps)
        else:
            rate = 1 - math.exp(-step / warmup)
        return rate

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    return scheduler


class MLPBlocks(nn.Module):
    def __init__(self, input_size, hidden_size, dropout, num_layers):
        super().__init__()
        self.activation = nn.ReLU()
        self.dropout_layer = nn.Dropout(p=dropout)
        self.input_layer = nn.Linear(input_size, hidden_size)
        middle_layer = nn.Linear(hidden_size, hidden_size)
        self.layers = get_clones(middle_layer, num_layers - 1)

    def forward(self, x):
        output = self.input_layer(x)
        output = self.dropout_layer(output)
        output = self.activation(output)
        old_output = output
        for layer in self.layers:
            output = layer(output)
            output = self.dropout_layer(output)
            output = self.activation(output) + old_output
            old_output = output
        return output


def get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def pad_packed_tensor(input, lengths, value):
    old_shape = input.shape
    device = input.device
    if not isinstance(lengths, torch.Tensor):
        lengths = torch.tensor(lengths, dtype=torch.int64, device=device)
    else:
        lengths = lengths.to(device)
    max_len = (lengths.max()).item()

    batch_size = len(lengths)
    x = input.new(batch_size * max_len, *old_shape[1:])
    x.fill_(value)

    index = torch.ones(len(input), dtype=torch.int64, device=device)
    row_shifts = torch.cumsum(max_len - lengths, 0)
    row_shifts_expanded = row_shifts[:-1].repeat_interleave(lengths[1:])

    cumsum_inds = torch.cumsum(index, 0) - 1
    cumsum_inds[lengths[0] :] += row_shifts_expanded
    x[cumsum_inds] = input
    return x.view(batch_size, max_len, *old_shape[1:])
