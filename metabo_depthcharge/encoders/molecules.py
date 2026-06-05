"""Molecule fingerprint encoders."""

import torch
import torch.nn as nn

from metabo_depthcharge.encoders.nn import AttnAggregator


class MolMLP(nn.Module):
    """Molecule embedder for a single fingerprint type.

    Projects a fingerprint vector to ``d_model`` via a linear layer followed
    by ``n_layers - 1`` residual blocks (LayerNorm → Linear → GELU). A
    normalisation step is applied at forward time depending on ``fp_type``:
    count FPs are normalised via dividing by ``max_counts`` vectors; dense FPs are L2-normalised;
    binary FPs are passed through unchanged.

    Given input :math:`x \\in \\mathbb{R}^{d_\\text{fp}}`, first apply
    type-specific normalisation:

    .. math::

        \\tilde{x} = \\text{norm}(x)

    Where :math:`\\text{norm}(\\cdot)` depends on ``fp_type`` as described above.
    then project and apply :math:`L - 1` residual blocks:

    .. math::

        h_0 = W_0\\,\\tilde{x} + b_0

    .. math::

        h_\\ell = h_{\\ell-1} + \\text{GELU}\\!\\left(
            W_\\ell\\,\\text{LN}(h_{\\ell-1}) + b_\\ell
        \\right), \\quad \\ell = 1,\\ldots,L-1

    where :math:`L` is ``n_layers``.

    Parameters
    ----------
    fp_size : int
        Input fingerprint dimensionality.
    n_layers : int
        Total depth. ``0`` → ``Identity`` (input passes through unchanged,
        so ``fp_size`` must equal ``d_model``); ``1`` → single ``Linear``;
        ``≥2`` → ``Linear`` + ``n_layers - 1`` residual blocks
        (``LayerNorm → Linear → GELU``).
    d_model : int, default 512
        Output embedding dimension.
    fp_type : str, default "binary"
        One of ``"binary"``, ``"count"``, or ``"dense"``.
    max_counts : torch.Tensor, optional
        ``(fp_size,)`` tensor of per-bit max counts used for count
        normalisation when ``fp_type="count"``. Registered as a buffer
        (not a learnable parameter). Obtain via
        :meth:`~metabo_depthcharge.datasets.molecules.MoleculeDataset.get_molmlp`
        to avoid manual derivation.
    """

    def __init__(
        self,
        fp_size: int,
        n_layers: int,
        d_model: int = 512,
        fp_type: str = "binary",
        max_counts: torch.Tensor = None,
    ):
        super().__init__()

        self.fp_type = fp_type

        if max_counts is not None:
            self.register_buffer("max_counts", max_counts)
        else:
            self.max_counts = None

        if n_layers == 0:
            self.init_layer = nn.Identity()
            self.blocks = None
        elif n_layers == 1:
            self.init_layer = nn.Linear(fp_size, d_model)
            self.blocks = None
        else:
            self.init_layer = nn.Linear(fp_size, d_model)
            self.blocks = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.LayerNorm(d_model),
                        nn.Linear(d_model, d_model),
                        nn.GELU(),
                    )
                    for _ in range(n_layers - 1)
                ]
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Embed a batch of fingerprint vectors.

        Applies type-specific normalisation before the MLP:
        count FPs are divided by ``max_counts``; dense FPs are L2-normalised;
        binary FPs are passed through unchanged.

        Parameters
        ----------
        x : torch.Tensor
            ``(B, fp_size)`` float tensor of fingerprint values.

        Returns
        -------
        torch.Tensor
            ``(B, d_model)`` float tensor of embeddings.
        """
        if self.fp_type == "count" and self.max_counts is not None:
            x = x / (self.max_counts + 1e-8)
        elif self.fp_type == "dense":
            x = torch.nn.functional.normalize(x, p=2, dim=-1)

        z = self.init_layer(x)
        if self.blocks is not None:
            for layer in self.blocks:
                z = z + layer(z)
        return z


class MultiMolMLP(nn.Module):
    """Embeds multiple fingerprint types and aggregates via attention pooling.

    Each fingerprint type gets its own :class:`MolMLP` projecting to
    ``d_model``. The per-type embeddings are stacked into
    ``(B, ..., N_fp, d_model)`` and aggregated with :class:`AttnAggregator`
    to produce ``(B, ..., d_model)``.

    Parameters
    ----------
    fp_names : list[str]
        Fingerprint names used as keys when indexing the input dict.
    fp_sizes : list[int]
        Input dimensionality for each fingerprint type.
    n_layers : int
        Number of layers passed to every :class:`MolMLP`.
    d_model : int, default 512
        Shared output embedding dimension.
    fp_types : list[str], optional
        ``fp_type`` for each fingerprint (defaults to ``"binary"`` for all
        if not provided).
    max_counts : dict[str, torch.Tensor], optional
        Dict mapping ``fp_name`` → max-counts tensor for count FPs.
    """

    def __init__(
        self,
        fp_names: list[str],
        fp_sizes: list[int],
        n_layers: int,
        d_model: int = 512,
        fp_types: list[str] = None,
        max_counts: dict[str, torch.Tensor] = None,
    ):
        super().__init__()
        self.fp_names = fp_names

        if fp_types is None:
            fp_types = ["binary"] * len(fp_names)
        if max_counts is None:
            max_counts = {}

        self.embedders = nn.ModuleDict(
            {
                name: MolMLP(
                    fp_size=size,
                    n_layers=n_layers,
                    d_model=d_model,
                    fp_type=fpt,
                    max_counts=max_counts.get(name),
                )
                for name, size, fpt in zip(fp_names, fp_sizes, fp_types, strict=False)
            }
        )

        self.aggregator = AttnAggregator(hidden_dim=d_model)

    def forward(self, fps: dict[str, torch.Tensor]) -> torch.Tensor:
        """Embed and aggregate multiple fingerprints.

        Parameters
        ----------
        fps : dict[str, torch.Tensor]
            Dict mapping ``fp_name`` → tensor of shape
            ``(B, ..., fp_size_i)``. All tensors must share leading dims.

        Returns
        -------
        torch.Tensor
            ``(B, ..., d_model)`` attention-aggregated embedding.
        """
        tokens = [self.embedders[name](fps[name]) for name in self.fp_names]
        stacked = torch.stack(tokens, dim=-2)  # (B, ..., N_fp, d_model)
        return self.aggregator(stacked)  # (B, ..., d_model)
