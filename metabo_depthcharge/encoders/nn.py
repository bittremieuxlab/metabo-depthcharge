"""General-purpose NN primitives shared across encoder families."""

import torch
import torch.nn as nn


class AttnAggregator(nn.Module):
    """Attention-weighted aggregation over a sequence dimension.

    Learns a scalar logit per position and softmax-normalises across the
    sequence, then returns the weighted sum. Works with any number of
    leading batch dimensions.

    Parameters
    ----------
    hidden_dim : int
        Dimensionality of the input features (D).
    """

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.to_attn_logits = nn.Linear(hidden_dim, 1)

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Aggregate over the second-to-last dimension.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (..., L, D).
        mask : torch.Tensor, optional
            Boolean mask of shape (..., L). ``True`` marks positions to
            exclude (they receive ``-inf`` before softmax).

        Returns
        -------
        torch.Tensor
            Aggregated tensor of shape (..., D).
        """
        attn_logits = self.to_attn_logits(x)  # ..., L, 1

        if mask is not None:
            attn_logits = attn_logits.masked_fill(mask.unsqueeze(-1), float("-inf"))

        attn_values = attn_logits.softmax(dim=-2)  # ..., L, 1

        return (x * attn_values).sum(-2)  # ..., D


class ResidualProjection(nn.Module):
    """Residual projection mapper from ``d_in`` to ``d_out``.

    Architecture:

    - **Projection**: ``Linear(d_in, d_out)`` when ``d_in != d_out``,
      else ``Identity``.
    - **Residual blocks** (only when ``n_layers > 0``): ``n_layers`` blocks
      of ``LayerNorm → Linear(d_out, 4*d_out) → GELU → Dropout →
      Linear(4*d_out, d_out)``, each added residually.
    - **Output norm**: ``LayerNorm(d_out)`` always applied last.

    Parameters
    ----------
    d_in : int
        Input feature dimension.
    d_out : int
        Output feature dimension.
    n_layers : int, default 0
        Number of residual blocks (0 = linear projection only).
    dropout : float, default 0.10
        Dropout rate inside each residual block.
    """

    def __init__(self, d_in: int, d_out: int, n_layers: int = 0, dropout: float = 0.10):
        super().__init__()

        if d_in != d_out:
            self.init_proj = nn.Linear(d_in, d_out)
        else:
            self.init_proj = nn.Identity()

        if n_layers > 0:
            blocks = []
            for _ in range(n_layers):
                block = nn.Sequential(
                    nn.LayerNorm(d_out),
                    nn.Linear(d_out, 4 * d_out),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(4 * d_out, d_out),
                )
                blocks.append(block)
            self.blocks = nn.ModuleList(blocks)
        else:
            self.blocks = None

        self.norm = nn.LayerNorm(d_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Project and optionally refine via residual blocks.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (..., d_in).

        Returns
        -------
        torch.Tensor
            Output tensor of shape (..., d_out).
        """
        x = self.init_proj(x)
        if self.blocks is not None:
            for block in self.blocks:
                x = x + block(x)
        return self.norm(x)
