"""Molecule fingerprint encoders."""

import torch
import torch.nn as nn

from metabo_depthcharge.encoders.nn import AttnAggregator, ResidualNetwork


class MolMLP(nn.Module):
    """Molecule embedder for a single representation.

    Given an input :math:`\\boldsymbol{x} \\in \\mathbb{R}^{d_\\text{rep}}`, a
    ``rep_type``-specific normalization is applied first: count
    fingerprints are divided by ``max_counts``, dense fingerprints are
    L2-normalized, and binary fingerprints are passed through unchanged. The
    normalized vector is then embedded by a
    :class:`ResidualNetwork` (see that class for the projection maths).
    Note that the :class:`ResidualNetwork` used here has ``final_norm=False``,
    so the output is not layer-normalized.

    Parameters
    ----------
    rep_size : int
        Input representation dimensionality.
    n_blocks : int
        Number of residual blocks in the internal :class:`ResidualNetwork`.
        ``0`` → projection only: a ``Linear`` to ``d_model``, or an exact pass-through when
        ``rep_size == d_model``.
        ``≥1`` adds that many residual blocks on top.
    d_model : int, default 512
        Output embedding dimension.
    rep_type : str, default "binary"
        One of ``"binary"``, ``"count"``, or ``"dense"``.
    max_counts : torch.Tensor, optional
        ``(rep_size,)`` shape tensor with per-dim max counts used for count
        normalization when ``rep_type="count"``. Obtain via
        :meth:`metabo_depthcharge.datasets.MoleculeDataset.get_molmlp`
        to avoid manual derivation.
    dropout : float, default 0.10
        Dropout rate inside the :class:`ResidualNetwork` residual blocks.
    """

    def __init__(
        self,
        rep_size: int,
        n_blocks: int,
        d_model: int = 512,
        rep_type: str = "binary",
        max_counts: torch.Tensor = None,
        dropout: float = 0.10,
    ):
        super().__init__()

        self.rep_type = rep_type

        if max_counts is not None:
            self.register_buffer("max_counts", max_counts)
        else:
            self.max_counts = None

        self.proj = ResidualNetwork(
            rep_size, d_model, n_blocks=n_blocks, dropout=dropout, final_norm=False
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Embed a batch of fingerprint vectors as described in the class docstring.

        Parameters
        ----------
        x : torch.Tensor
            ``(B, rep_size)`` float tensor of fingerprint values.

        Returns
        -------
        torch.Tensor
            ``(B, d_model)`` float tensor of embeddings.
        """
        if self.rep_type == "count" and self.max_counts is not None:
            x = x / (self.max_counts + 1e-8)
        elif self.rep_type == "dense":
            x = torch.nn.functional.normalize(x, p=2, dim=-1)

        return self.proj(x)


class MultiMolMLP(nn.Module):
    """Embeds multiple representation types and aggregates via attention pooling.

    Each representation type gets its own :class:`MolMLP` projecting to
    ``d_model``. The per-type embeddings are stacked into
    ``(B, ..., N_fp, d_model)`` and aggregated with :class:`AttnAggregator`
    to produce ``(B, ..., d_model)``.

    Parameters
    ----------
    rep_names : list[str]
        Representation names used as keys when indexing the input dict.
    rep_sizes : list[int]
        Input dimensionality for each representation type.
    n_blocks : int
        Number of residual blocks passed to every :class:`MolMLP`.
    d_model : int, default 512
        Shared output embedding dimension.
    rep_types : list[str], optional
        ``rep_type`` for each representation (defaults to ``"binary"`` for all
        if not provided).
    max_counts : dict[str, torch.Tensor], optional
        Dict mapping ``rep_name`` → max-counts tensor for count FPs.
    dropout : float, default 0.10
        Dropout rate passed to every :class:`MolMLP`.
    """

    def __init__(
        self,
        rep_names: list[str],
        rep_sizes: list[int],
        n_blocks: int,
        d_model: int = 512,
        rep_types: list[str] = None,
        max_counts: dict[str, torch.Tensor] = None,
        dropout: float = 0.10,
    ):
        super().__init__()
        self.rep_names = rep_names

        if rep_types is None:
            rep_types = ["binary"] * len(rep_names)
        if max_counts is None:
            max_counts = {}

        self.embedders = nn.ModuleDict(
            {
                name: MolMLP(
                    rep_size=size,
                    n_blocks=n_blocks,
                    d_model=d_model,
                    rep_type=fpt,
                    max_counts=max_counts.get(name),
                    dropout=dropout,
                )
                for name, size, fpt in zip(
                    rep_names, rep_sizes, rep_types, strict=False
                )
            }
        )

        self.aggregator = AttnAggregator(hidden_dim=d_model)

    def forward(self, reps: dict[str, torch.Tensor]) -> torch.Tensor:
        """Embed and aggregate multiple representations.

        Parameters
        ----------
        reps : dict[str, torch.Tensor]
            Dict mapping ``rep_name`` → tensor of shape
            ``(B, ..., rep_size_i)``. All tensors must share leading dims.

        Returns
        -------
        torch.Tensor
            ``(B, ..., d_model)``. Attention-aggregated embedding.
        """
        tokens = [self.embedders[name](reps[name]) for name in self.rep_names]
        stacked = torch.stack(tokens, dim=-2)  # (B, ..., N_fp, d_model)
        return self.aggregator(stacked)  # (B, ..., d_model)
