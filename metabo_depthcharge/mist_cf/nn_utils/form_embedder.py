# Adapted from MIST-CF (Goldman et al., 2023)
# Licensed under MIT License - see LICENSE in this directory
"""form_embedder.py"""

import numpy as np
import torch
import torch.nn as nn

from .. import common


class IntFeaturizer(nn.Module):
    MAX_COUNT_INT = 255
    NUM_EXTRA_EMBEDDINGS = 1

    def __init__(self, embedding_dim):
        super().__init__()
        weights = torch.zeros(self.NUM_EXTRA_EMBEDDINGS, embedding_dim)
        self._extra_embeddings = nn.Parameter(weights, requires_grad=True)
        nn.init.normal_(self._extra_embeddings, 0.0, 1.0)
        self.embedding_dim = embedding_dim

    def forward(self, tensor):
        orig_shape = tensor.shape
        out_tensor = torch.empty(
            (*orig_shape, self.embedding_dim), device=tensor.device
        )
        extra_embed = tensor >= self.MAX_COUNT_INT

        tensor = tensor.long()
        norm_embeds = self.int_to_feat_matrix[tensor[~extra_embed]]
        extra_embeds = self._extra_embeddings[tensor[extra_embed] - self.MAX_COUNT_INT]

        out_tensor[~extra_embed] = norm_embeds
        out_tensor[extra_embed] = extra_embeds

        temp_out = out_tensor.reshape(*orig_shape[:-1], -1)
        return temp_out

    @property
    def num_dim(self):
        return self.int_to_feat_matrix.shape[1]

    @property
    def full_dim(self):
        return self.num_dim * common.NORM_VEC.shape[0]


class FourierFeaturizer(IntFeaturizer):
    def __init__(self):
        num_freqs = int(np.ceil(np.log2(self.MAX_COUNT_INT))) + 2
        freqs = 0.5 ** torch.arange(num_freqs, dtype=torch.float32)
        freqs_time_2pi = 2 * np.pi * freqs
        super().__init__(embedding_dim=2 * freqs_time_2pi.shape[0])

        combo_of_sinusoid_args = (
            torch.arange(self.MAX_COUNT_INT, dtype=torch.float32)[:, None]
            * freqs_time_2pi[None, :]
        )
        all_features = torch.cat(
            [torch.cos(combo_of_sinusoid_args), torch.sin(combo_of_sinusoid_args)],
            dim=1,
        )
        self.int_to_feat_matrix = nn.Parameter(all_features.float())
        self.int_to_feat_matrix.requires_grad = False


class FourierFeaturizerSines(IntFeaturizer):
    def __init__(self):
        num_freqs = int(np.ceil(np.log2(self.MAX_COUNT_INT))) + 2
        freqs = (0.5 ** torch.arange(num_freqs, dtype=torch.float32))[2:]
        freqs_time_2pi = 2 * np.pi * freqs
        super().__init__(embedding_dim=freqs_time_2pi.shape[0])

        combo_of_sinusoid_args = (
            torch.arange(self.MAX_COUNT_INT, dtype=torch.float32)[:, None]
            * freqs_time_2pi[None, :]
        )
        self.int_to_feat_matrix = nn.Parameter(
            torch.sin(combo_of_sinusoid_args).float()
        )
        self.int_to_feat_matrix.requires_grad = False


class FourierFeaturizerAbsoluteSines(IntFeaturizer):
    def __init__(self):
        num_freqs = int(np.ceil(np.log2(self.MAX_COUNT_INT))) + 2
        freqs = (0.5 ** torch.arange(num_freqs, dtype=torch.float32))[2:]
        freqs_time_2pi = 2 * np.pi * freqs
        super().__init__(embedding_dim=freqs_time_2pi.shape[0])

        combo_of_sinusoid_args = (
            torch.arange(self.MAX_COUNT_INT, dtype=torch.float32)[:, None]
            * freqs_time_2pi[None, :]
        )
        self.int_to_feat_matrix = nn.Parameter(
            torch.abs(torch.sin(combo_of_sinusoid_args)).float()
        )
        self.int_to_feat_matrix.requires_grad = False


class RBFFeaturizer(IntFeaturizer):
    def __init__(self, num_funcs=32):
        super().__init__(embedding_dim=num_funcs)
        width = (self.MAX_COUNT_INT - 1) / num_funcs
        centers = torch.linspace(0, self.MAX_COUNT_INT - 1, num_funcs)

        pre_exponential_terms = (
            -0.5
            * ((torch.arange(self.MAX_COUNT_INT)[:, None] - centers[None, :]) / width)
            ** 2
        )
        feats = torch.exp(pre_exponential_terms)
        self.int_to_feat_matrix = nn.Parameter(feats.float())
        self.int_to_feat_matrix.requires_grad = False


class OneHotFeaturizer(IntFeaturizer):
    def __init__(self):
        super().__init__(embedding_dim=self.MAX_COUNT_INT)
        feats = torch.eye(self.MAX_COUNT_INT)
        self.int_to_feat_matrix = nn.Parameter(feats.float())
        self.int_to_feat_matrix.requires_grad = False


class LearnedFeaturizer(IntFeaturizer):
    def __init__(self, feature_dim=32):
        super().__init__(embedding_dim=feature_dim)
        weights = torch.zeros(self.MAX_COUNT_INT, feature_dim)
        self.int_to_feat_matrix = nn.Parameter(weights, requires_grad=True)
        nn.init.normal_(self.int_to_feat_matrix, 0.0, 1.0)


class FloatFeaturizer(IntFeaturizer):
    def __init__(self):
        super().__init__(embedding_dim=1)
        self.norm_vec = torch.from_numpy(common.NORM_VEC).float()
        self.norm_vec = nn.Parameter(self.norm_vec)
        self.norm_vec.requires_grad = False

    def forward(self, tensor):
        tens_shape = tensor.shape
        out_shape = [1] * (len(tens_shape) - 1) + [-1]
        return tensor / self.norm_vec.reshape(*out_shape)

    @property
    def num_dim(self):
        return 1


def get_embedder(embedder):
    if embedder == "fourier":
        embedder = FourierFeaturizer()
    elif embedder == "rbf":
        embedder = RBFFeaturizer()
    elif embedder == "one-hot":
        embedder = OneHotFeaturizer()
    elif embedder == "learnt":
        embedder = LearnedFeaturizer()
    elif embedder == "float":
        embedder = FloatFeaturizer()
    elif embedder == "fourier-sines":
        embedder = FourierFeaturizerSines()
    elif embedder == "abs-sines":
        embedder = FourierFeaturizerAbsoluteSines()
    else:
        raise NotImplementedError(f"Unknown embedder: {embedder}")
    return embedder
