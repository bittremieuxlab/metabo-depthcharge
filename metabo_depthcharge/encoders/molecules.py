"""Molecule encoders: fingerprint MLPs and graph neural networks."""

from collections.abc import Sequence

import torch
import torch.nn as nn
from torch.nn import functional

from metabo_depthcharge.chem import graphs
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


class GraphMolEncoder(nn.Module):
    r"""GCN over molecular graphs, trained from scratch.

    Each atom updates its representation by summing its neighbors' (plus its own)
    current representations, passing that sum through a linear layer, and adding a
    separate linear "skip" pass-through of the atom's own old representation. Both
    halves go through ReLU before being added, then the whole thing is normalized:

    .. math::

        h_i \leftarrow \mathrm{norm}\Big(
            \mathrm{relu}\big(\textstyle\sum_{j \in N(i) \cup \{i\}} x_j W + b\big)
            + \mathrm{relu}(x_i W_{res} + b_{res})
        \Big)

    Dropout is applied before the norm. This repeats for each entry in ``channels``,
    so a molecule's final atom representations depend on progressively larger
    neighborhoods the deeper the stack goes.

    Note that the neighbor sum is a plain sum (equivalent to dgllife's
    ``gnn_norm='none'``). An atom's own value is folded into
    that same sum through self-loops.

    Parameters
    ----------
    atom_types : torch.Tensor
        ``(n_types, FEAT_DIM)`` table of distinct atom feature rows, from
        :meth:`~metabo_depthcharge.datasets.MoleculeDataset.atom_types`.
    channels : sequence of int
        Output width of each graph-convolution layer.
    dropout : float
        Dropout within each layer.
    d_model : int
        Output dimension. When it equals ``channels[-1]`` no projection is added.
    per_atom : bool, default False
        If ``True``, return one token per ATOM together with a padding mask.
        If ``False``, mean-pool the graph to a single vector per molecule.
    norm : {"batch", "layer"}, default "batch"
        Normalization after each layer.
    """

    def __init__(
        self,
        atom_types: torch.Tensor,
        channels: Sequence[int],
        dropout: float,
        d_model: int,
        per_atom: bool = False,
        norm: str = "batch",
    ):
        super().__init__()
        if norm not in ("batch", "layer"):
            raise ValueError(f"Unknown norm: {norm}")
        self.per_atom = per_atom
        self.register_buffer("atom_types", torch.as_tensor(atom_types).float())

        dims = [graphs.FEAT_DIM, *channels]
        self.convs = nn.ModuleList(
            nn.Linear(i, o) for i, o in zip(dims, dims[1:], strict=False)
        )
        self.res = nn.ModuleList(
            nn.Linear(i, o) for i, o in zip(dims, dims[1:], strict=False)
        )
        self.bns = nn.ModuleList(
            (nn.BatchNorm1d(o) if norm == "batch" else nn.LayerNorm(o))
            for o in channels
        )
        self.drop = nn.Dropout(dropout)
        self.out = (
            nn.Identity()
            if channels[-1] == d_model
            else nn.Linear(channels[-1], d_model)
        )

    def _unpack(self, graph):
        """A batched graph -> atom rows and batch-local edges."""
        ids = graph["atom_type"].long()
        if ids.numel() and int(ids.max()) >= len(self.atom_types):
            raise ValueError(
                "batch has an atom type id outside this encoder's table; the table "
                "and the graphs must come from the same dataset"
            )
        device = self.atom_types.device
        nsize = torch.diff(graph["nptr"]).to(device)
        bsize = torch.diff(graph["bptr"]).to(device)
        src, dst, ecode, erev = graphs.expand_bonds(
            nsize,
            bsize,
            graph["bsrc"].to(device),
            graph["bdst"].to(device),
            graph["bcode"].to(device),
        )
        return self.atom_types[ids.to(device)], nsize, bsize, src, dst, ecode, erev

    def forward(self, graph: dict):
        """Encode a batch of molecular graphs.

        Parameters
        ----------
        graph : dict
            A batched graph from
            :meth:`~metabo_depthcharge.datasets.MoleculeDataset.gather_graphs` or
            :func:`~metabo_depthcharge.chem.graphs.collate`.

        Returns
        -------
        torch.Tensor or tuple[torch.Tensor, torch.Tensor]
            ``(B, d_model)`` with `B=batch_size`, when ``per_atom`` is ``False``.
            Otherwise, a ``((B, max_atoms, d_model), (B, max_atoms))`` tuple of
            per-atom tokens and their padding mask.
        """
        x, nsize, _, src, dst, _, _ = self._unpack(graph)
        n = x.shape[0]
        adj = torch.sparse_coo_tensor(
            torch.stack([dst, src]), torch.ones(len(src), device=x.device), (n, n)
        ).coalesce()

        for conv, res, bn in zip(self.convs, self.res, self.bns, strict=True):
            agg = torch.sparse.mm(adj, functional.linear(x, conv.weight)) + conv.bias
            x = bn(self.drop(functional.relu(agg) + functional.relu(res(x))))

        return self._scatter(self.out(x), nsize)

    def _scatter(self, x, nsize):
        """Node states -> per-molecule output, pooled or padded-with-mask."""
        n_graphs = len(nsize)
        graph_id = torch.repeat_interleave(
            torch.arange(n_graphs, device=x.device), nsize
        )
        if not self.per_atom:
            pooled = torch.zeros(n_graphs, x.shape[1], device=x.device, dtype=x.dtype)
            pooled.index_add_(0, graph_id, x)
            return pooled / nsize[:, None].to(x.dtype)
        local = torch.arange(x.shape[0], device=x.device) - torch.repeat_interleave(
            torch.cumsum(nsize, 0) - nsize, nsize
        )
        width = int(nsize.max())
        out = torch.zeros(n_graphs, width, x.shape[1], device=x.device, dtype=x.dtype)
        out[graph_id, local] = x
        pad = torch.arange(width, device=x.device)[None, :] >= nsize[:, None]
        return out, pad


class BondMolEncoder(GraphMolEncoder):
    r"""D-MPNN: directed bond-level message passing on molecular graphs.

    Where a node-centered GNN keeps a hidden state per atom and mixes over neighbor
    atoms, D-MPNN keeps a hidden state per (directed) bond and mixes over incoming
    bonds, deliberately excluding its own bond in the opposite direction:

    .. math::

        m_{vw}^{(0)} &= \mathrm{relu}\big(W_0 [\, x_v \,\|\, e_{vw} \,]\big) \\
        m_{vw}^{(t+1)} &= \mathrm{relu}\Big(
            m_{vw}^{(0)} + W_h \Big(\textstyle\sum_{k \in N(v)} m_{kv}^{(t)} - m_{wv}^{(t)}\Big)
        \Big) \\
        h_v &= \mathrm{relu}\Big(W_o \big[\, x_v \,\|\, \textstyle\sum_{k \in N(v)} m_{kv}^{(T)} \,\big]\Big)

    :math:`m_{vw}` is the message living on the directed bond :math:`v \to w`
    (initialized from the source atom's features :math:`x_v` and the bond's type
    embedding :math:`e_{vw}`). :math:`N(v)` is atom :math:`v`'s neighbors; and
    :math:`m_{wv}`, the reverse of :math:`m_{vw}` itself, is subtracted out of the
    sum before it updates.

    Parameters
    ----------
    atom_types : torch.Tensor
        ``(n_types, FEAT_DIM)`` atom-feature table; see :class:`GraphMolEncoder`.
    channels : sequence of int
        Output width of each message-passing layer.
    dropout : float
        Dropout within each layer.
    d_model : int
        Output dimension. When it equals ``channels[-1]`` no projection is added.
    per_atom : bool, default False
        If ``True``, return one token per ATOM together with a padding mask.
        If ``False``, mean-pool the graph to a single vector per molecule.
    norm : {"batch", "layer"}, default "batch"
        Normalization after each layer.
    bond_tokens : bool, default False
        Emit one token per undirected bond instead of per atom.
    """

    def __init__(
        self,
        atom_types: torch.Tensor,
        channels: Sequence[int],
        dropout: float,
        d_model: int,
        per_atom: bool = False,
        norm: str = "batch",
        bond_tokens: bool = False,
    ):
        super().__init__(
            atom_types, channels, dropout, d_model, per_atom=per_atom, norm=norm
        )
        self.convs, self.res, self.bns = (nn.ModuleList() for _ in range(3))
        self.bond_tokens = bond_tokens
        self.ecodes = nn.ModuleList([nn.Embedding(graphs.N_BOND_CODES, channels[0])])
        h = channels[0]
        self.h0 = nn.Linear(graphs.FEAT_DIM + h, h)
        self.mmsg = nn.Linear(h, h)
        self.node_out = nn.Linear(graphs.FEAT_DIM + h, channels[-1])
        self.depth = len(channels)
        if bond_tokens:
            self.bond_out = nn.Linear(h, channels[-1])

    def _propagate(self, x, src, dst, ecode, erev, n):
        """Node states -> (updated node states, directed-edge messages)."""
        h0 = functional.relu(
            self.h0(torch.cat([x[src], self.ecodes[0](ecode)], dim=-1))
        )
        m = h0
        for _ in range(self.depth - 1):
            node_sum = torch.zeros(n, m.shape[1], device=m.device, dtype=m.dtype)
            node_sum.index_add_(0, dst, m)
            m = functional.relu(h0 + self.mmsg(node_sum[src] - m[erev]))
            m = self.drop(m)
        node_sum = torch.zeros(n, m.shape[1], device=m.device, dtype=m.dtype)
        node_sum.index_add_(0, dst, m)
        out = functional.relu(self.node_out(torch.cat([x, node_sum], dim=-1)))
        return out, m

    def forward(self, graph: dict):
        """Encode a batch of graphs; see :meth:`GraphMolEncoder.forward`."""
        x, nsize, bsize, src, dst, ecode, erev = self._unpack(graph)
        x, m = self._propagate(x, src, dst, ecode, erev, x.shape[0])
        if self.bond_tokens:
            return self._bond_scatter(m, ecode, erev, bsize)
        return self._scatter(self.out(x), nsize)

    def _bond_scatter(self, m, ecode, erev, bsize):
        """Directed-edge states -> one token per undirected bond, plus a padding mask."""
        keep = ecode != graphs.SELF_LOOP_CODE
        first = keep & (torch.arange(len(erev), device=m.device) < erev)
        tok = self.out(self.bond_out(m[first] + m[erev[first]]))

        graph_id = torch.repeat_interleave(
            torch.arange(len(bsize), device=m.device), bsize
        )
        local = torch.arange(len(graph_id), device=m.device) - torch.repeat_interleave(
            torch.cumsum(bsize, 0) - bsize, bsize
        )
        width = int(bsize.max().clamp(min=1))
        out = torch.zeros(
            len(bsize), width, tok.shape[1], device=m.device, dtype=tok.dtype
        )
        out[graph_id, local] = tok
        pad = torch.arange(width, device=m.device)[None, :] >= bsize[:, None]
        return out, pad
