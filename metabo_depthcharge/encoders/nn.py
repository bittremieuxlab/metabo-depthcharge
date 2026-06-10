"""General-purpose NN primitives shared across encoder families."""

import torch
import torch.nn as nn


class AttnAggregator(nn.Module):
    """Attention-weighted sum over a sequence dimension.

    A simple learned alternative for mean or max pooling.
    Learns a scalar logit per position and softmax-normalizes across the
    sequence, then returns the weighted sum. Works with any number of
    leading batch dimensions.

    For a sequence :math:`(\\boldsymbol{x}_1, \\ldots, \\boldsymbol{x}_L)` with
    :math:`\\boldsymbol{x}_i \\in \\mathbb{R}^D`, a learned linear scorer assigns
    each position a logit :math:`e_i = \\boldsymbol{w}^\\top \\boldsymbol{x}_i +
    b`, which is softmax-normalized into attention weights and used to take a
    weighted sum of the inputs:

    .. math::

        \\alpha_i = \\frac{\\exp(e_i)}{\\sum_{j=1}^{L} \\exp(e_j)},
        \\qquad
        \\boldsymbol{z} = \\sum_{i=1}^{L} \\alpha_i \\, \\boldsymbol{x}_i

    Positions flagged by ``mask`` are assigned :math:`e_i = -\\infty` before the
    softmax, so their weight :math:`\\alpha_i = 0` and they do not contribute to
    :math:`\\boldsymbol{z}`.

    Parameters
    ----------
    hidden_dim : int
        Dimensionality of the input features (:math:`D`).
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
            Input tensor of shape ``(..., L, D)``.
        mask : torch.Tensor, optional
            Boolean mask of shape ``(..., L)``. ``True`` marks positions to
            exclude (they receive ``-inf`` before softmax).

        Returns
        -------
        torch.Tensor
            Aggregated tensor of shape ``(..., D)``.
        """
        attn_logits = self.to_attn_logits(x)  # ..., L, 1

        if mask is not None:
            attn_logits = attn_logits.masked_fill(mask.unsqueeze(-1), float("-inf"))

        attn_values = attn_logits.softmax(dim=-2)  # ..., L, 1

        return (x * attn_values).sum(-2)  # ..., D


class ResidualNetwork(nn.Module):
    """Residual projection mapper from ``d_in`` to ``d_out``.

    Architecture:

    - **Projection**: ``Linear(d_in, d_out)`` when ``d_in != d_out``,
      else ``Identity``.
    - **Residual blocks** (only when ``n_blocks > 0``): ``n_blocks`` blocks
      of ``LayerNorm → Linear(d_out, 4*d_out) → GELU → Dropout →
      Linear(4*d_out, d_out)``, each added residually.
    - **Output norm**: ``LayerNorm(d_out)`` applied last, unless
      ``final_norm=False`` (then ``Identity``).

    In maths, given an input :math:`\\boldsymbol{x} \\in \\mathbb{R}^{d_\\text{in}}`, an
    initial projection maps it to :math:`d_\\text{out}` dimensions,

    .. math::

        \\boldsymbol{h}_0 = \\begin{cases}
            W_0\\,\\boldsymbol{x} + \\boldsymbol{b}_0
                & d_\\text{in} \\neq d_\\text{out}, \\\\
            \\boldsymbol{x} & d_\\text{in} = d_\\text{out},
        \\end{cases}

    followed by :math:`L` (``n_blocks``) residual blocks

    .. math::

        \\boldsymbol{h}_\\ell = \\boldsymbol{h}_{\\ell-1}
        + W_\\ell^{(2)}\\,\\operatorname{GELU}\\!\\left(
            W_\\ell^{(1)}\\,\\operatorname{LN}(\\boldsymbol{h}_{\\ell-1})
        \\right),
        \\qquad \\ell = 1, \\ldots, L,

    with :math:`W_\\ell^{(1)} \\in \\mathbb{R}^{4 d_\\text{out} \\times
    d_\\text{out}}` and :math:`W_\\ell^{(2)} \\in \\mathbb{R}^{d_\\text{out}
    \\times 4 d_\\text{out}}`. When ``final_norm=True`` a final layer norm
    gives the output :math:`\\boldsymbol{y} = \\operatorname{LN}(\\boldsymbol{h}_L)`;
    otherwise :math:`\\boldsymbol{y} = \\boldsymbol{h}_L`. Biases and dropout
    inside the blocks above are omitted for clarity.

    Parameters
    ----------
    d_in : int
        Input feature dimension.
    d_out : int
        Output feature dimension.
    n_blocks : int, default 0
        Number of residual blocks (0 = linear projection only).
    dropout : float, default 0.10
        Dropout rate inside each residual block.
    final_norm : bool, default True
        Whether to apply a ``LayerNorm`` to the output. Set ``False`` for an
        un-normalised output (e.g. so that ``n_blocks=0`` with
        ``d_in == d_out`` is an exact pass-through).
    """

    def __init__(
        self,
        d_in: int,
        d_out: int,
        n_blocks: int = 0,
        dropout: float = 0.10,
        final_norm: bool = True,
    ):
        super().__init__()

        if d_in != d_out:
            self.init_proj = nn.Linear(d_in, d_out)
        else:
            self.init_proj = nn.Identity()

        if n_blocks > 0:
            blocks = []
            for _ in range(n_blocks):
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

        self.norm = nn.LayerNorm(d_out) if final_norm else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Pass inputs through residual projections as described in class docstring.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape ``(..., d_in)``.

        Returns
        -------
        torch.Tensor
            Output tensor of shape ``(..., d_out)``.
        """
        x = self.init_proj(x)
        if self.blocks is not None:
            for block in self.blocks:
                x = x + block(x)
        return self.norm(x)
