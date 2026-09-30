import warnings
from collections.abc import Sequence

import torch
import torch.nn as nn
from depthcharge.encoders import FloatEncoder
from depthcharge.transformers import SpectrumTransformerEncoder
from torch.nn import functional

from metabo_depthcharge.encoders.nn import AttnAggregator, GrowableEmbedding
from metabo_depthcharge.mist_cf.common.chem_utils import VALID_ELEMENTS
from metabo_depthcharge.mist_cf.nn_utils import get_embedder
from metabo_depthcharge.spec.adducts import N_ADDUCTS
from metabo_depthcharge.spec.metadata_parsers import (
    N_INSTRUMENTS,
    N_ION_ACTIVATIONS,
    N_IONIZATION_METHODS,
)


class MetadataEncoder(nn.Module):
    """Encode spectrum acquisition metadata into a ``d_model`` vector.

    Encodes the fields of an input metadata dictionary, each with its own encoder:
    The per-field outputs are summed into a single ``(B, d_model)`` vector.

    - ``"adduct"`` — a categorical index into the adduct vocabulary.
      See :func:`~metabo_depthcharge.spec.adducts.encode_adduct` for encoding
      adducts to integers.
      Embedded via :class:`torch.nn.Embedding`.
      Unknown adducts should be passed as 0, which is assigned
      a zero embedding.
    - ``"instrument_type"`` — a categorical index into the instrument
      vocabulary. See
      :func:`~metabo_depthcharge.spec.metadata_parsers.encode_instrument` for
      encoding instruments to integers.
      Embedded via :class:`torch.nn.Embedding`.
      Unknown instruments should be passed as 0, which is assigned
      a zero embedding.
    - ``"collision_energy"`` — a single continuous value. See
      :func:`~metabo_depthcharge.spec.metadata_parsers.encode_collision_energy`
      for parsing raw values to the float consumed here.
      Projected from one dimension to ``d_model`` by a
      :class:`torch.nn.Linear`. A value of ``0.0`` is treated as missing and
      masked back to zeros after the projection (the bias would otherwise
      leak a nonzero embedding).
    - ``"ion_activation"`` — a categorical index into the ion activation
      vocabulary (e.g. HCD, CID). See
      :func:`~metabo_depthcharge.spec.metadata_parsers.encode_ion_activation`.
      Embedded via :class:`torch.nn.Embedding`.
      Unknown methods should be passed as 0, which is assigned a zero embedding.
    - ``"ionization_method"`` — a categorical index into the ionization method
      vocabulary (e.g. ESI, NSI, APCI). See
      :func:`~metabo_depthcharge.spec.metadata_parsers.encode_ionization_method`.
      Embedded via :class:`torch.nn.Embedding`.
      Unknown methods should be passed as 0, which is assigned a zero embedding.

    Missing values (index 0 for the categoricals, ``0.0`` for collision
    energy) therefore contribute zero to the sum for that sample, and only
    the fields listed in ``metadata_fields`` are instantiated.

    Parameters
    ----------
    d_model : int
        Output embedding dimension.
    metadata_fields : list[str]
        Field names to encode; subset of
        ``["adduct", "collision_energy", "instrument_type", "ion_activation", "ionization_method"]``.
    """

    def __init__(self, d_model: int, metadata_fields: list[str]):
        super().__init__()
        if len(metadata_fields) == 0:
            raise ValueError("At least one metadata field must be enabled")
        self.metadata_fields = metadata_fields

        if "adduct" in metadata_fields:
            # Index 0 = unknown/other, padding_idx=0 so unknown → zero vector.
            self.adduct_emb = GrowableEmbedding(N_ADDUCTS, d_model, padding_idx=0)

        if "collision_energy" in metadata_fields:
            self.ce_encoder = nn.Linear(1, d_model)

        if "instrument_type" in metadata_fields:
            self.instrument_emb = GrowableEmbedding(
                N_INSTRUMENTS, d_model, padding_idx=0
            )

        if "ion_activation" in metadata_fields:
            self.ion_activation_emb = GrowableEmbedding(
                N_ION_ACTIVATIONS, d_model, padding_idx=0
            )

        if "ionization_method" in metadata_fields:
            self.ionization_method_emb = GrowableEmbedding(
                N_IONIZATION_METHODS, d_model, padding_idx=0
            )

    def forward(self, metadata: dict[str, torch.Tensor]) -> torch.Tensor:
        """Encode metadata fields into a summed embedding.

        Parameters
        ----------
        metadata : dict[str, torch.Tensor]
            Dict mapping field names to tensors. ``"adduct"`` and
            ``"instrument_type"`` are ``(B,)`` int64 indices;
            ``"collision_energy"`` is a ``(B,)`` float value.
            Fields absent from the dict contribute zero.

        Returns
        -------
        torch.Tensor
            ``(B, d_model)`` float tensor — sum of all enabled field embeddings.
        """
        parts = []

        if "adduct" in self.metadata_fields and "adduct" in metadata:
            parts.append(self.adduct_emb(metadata["adduct"].long()))

        if (
            "collision_energy" in self.metadata_fields
            and "collision_energy" in metadata
        ):
            ce = metadata["collision_energy"].to(self.ce_encoder.weight.dtype)
            ce_emb = self.ce_encoder(ce[:, None])  # (B, d_model)
            mask = (ce != 0.0).unsqueeze(-1)  # (B, 1)
            parts.append(ce_emb * mask)

        if "instrument_type" in self.metadata_fields and "instrument_type" in metadata:
            parts.append(self.instrument_emb(metadata["instrument_type"].long()))

        if "ion_activation" in self.metadata_fields and "ion_activation" in metadata:
            parts.append(self.ion_activation_emb(metadata["ion_activation"].long()))

        if (
            "ionization_method" in self.metadata_fields
            and "ionization_method" in metadata
        ):
            parts.append(
                self.ionization_method_emb(metadata["ionization_method"].long())
            )

        if not parts:
            warnings.warn(
                f"MetadataEncoder has fields {self.metadata_fields} enabled, "
                "but none were present in the forward() input; contributing zeros.",
                stacklevel=2,
            )
            return 0

        return sum(parts)


class PeakEncoder(nn.Module):
    """Encode (m/z, intensity) peak pairs into ``d_model``-dimensional vectors.

    Differs from depthcharge's default
    `PeakEncoder <https://wfondrie.github.io/depthcharge/latest/api/encoders/#depthcharge.encoders.PeakEncoder>`_: instead of
    concatenating the m/z and intensity encodings and projecting, it sums
    them, matching the original transformer formulation of Vaswani et al.
    [1]_.

    Parameters
    ----------
    d_model : int
        Output embedding dimension.
    min_mz_wavelength : float
        Minimum wavelength for m/z sinusoidal encoding.
    max_mz_wavelength : float
        Maximum wavelength for m/z sinusoidal encoding.
    use_mz : bool, default True
        Include the m/z encoding in the summed output. ``False`` means only intensity
        is encoded.
    use_intensity : bool, default True
        Include the sinusoidal intensity encoding in the summed output. ``False``
        means intensity contributes nothing here -- use this when intensity is
        instead folded into a :class:`SubformulaEncoder` with
        ``include_intensity=True`` (see :class:`SpectrumEncoder`'s
        ``use_intensity`` flag).

    References
    ----------
    .. [1] Vaswani, Ashish, et al. "Attention Is All You
       Need." Advances in Neural Information Processing Systems 30 (2017).
       https://arxiv.org/abs/1706.03762
    """

    def __init__(
        self,
        d_model: int,
        min_mz_wavelength: float,
        max_mz_wavelength: float,
        use_mz: bool = True,
        use_intensity: bool = True,
    ):
        super().__init__()
        self.d_model = d_model
        self.use_mz = use_mz
        self.use_intensity = use_intensity

        self.mz_encoder = FloatEncoder(
            d_model=self.d_model,
            min_wavelength=min_mz_wavelength,
            max_wavelength=max_mz_wavelength,
            learnable_wavelengths=False,
        )

        self.int_encoder = FloatEncoder(
            d_model=self.d_model,
            min_wavelength=1e-6,
            max_wavelength=1,
            learnable_wavelengths=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode a batch of peak sequences.

        Note that we expect intensities to fall within the interval ``[0, 1]``.

        Parameters
        ----------
        x : torch.Tensor
            ``(B, L, 2)`` tensor with m/z values in ``x[..., 0]`` and
            intensities in ``x[..., 1]``.

        Returns
        -------
        torch.Tensor
            ``(B, L, d_model)`` float tensor.
        """
        out = self.int_encoder(x[:, :, 1]) if self.use_intensity else 0
        if self.use_mz:
            out = out + self.mz_encoder(x[:, :, 0])
        return out


class SpectrumEncoder(SpectrumTransformerEncoder):
    """Spectrum encoder based on depthcharge's ``SpectrumTransformerEncoder``.

    Wraps `SpectrumTransformerEncoder
    <https://wfondrie.github.io/depthcharge/latest/api/transformers/#depthcharge.transformers.SpectrumTransformerEncoder>`_
    + optional subformulae, metadata, and pooling into a single module.
    Note that the ``dim_feedforward`` argument of the underlying transformer is fixed to
    ``d_model * 4``, following the original transformer architecture.

    Parameters
    ----------
    d_model : int, default 512
        Transformer hidden dimension (also the output dimension).
    n_layers : int, default 8
        Number of transformer layers.
    nhead : int, default 8
        Number of attention heads per layer; ``d_model`` must be divisible
        by ``nhead``.
    dropout : float, default 0.15
        Dropout rate.
    min_mz_wavelength : float, default 0.001
        Min wavelength for m/z positional encoding.
    max_mz_wavelength : float, default 10000
        Max wavelength for m/z positional encoding.
    pool : str or None, default "attention"
        Pooling applied to the transformer output: ``"attention"`` (learned
        weighted sum over tokens, i.e. :class:`AttnAggregator`), ``"cls"``
        (global/first token), ``"last"`` (hidden state at each sample's last
        non-padded position), or ``None`` to skip pooling and return the full
        token sequence together with its padding mask.
    subformula_encoder : nn.Module, optional
        :class:`SubformulaEncoder` whose output is added to peak embeddings
        before the transformer.
    metadata_encoder : nn.Module, optional
        :class:`MetadataEncoder` whose output is added to the global/CLS
        token before the transformer. Requires ``use_global_token=True``.
    use_mz : bool, default True
        Whether to include sinusoidal m/z in the peak embeddings and precursor token.
        Set ``False`` to build peak tokens from intensity and formulae alone, so that
        m/z never enters as a number. Requires a ``subformula_encoder``.
    intensity_encoding : {"sinusoidal", "joint_formula"}, default "sinusoidal"
        How intensity enters the peak embeddings. ``"sinusoidal"`` (default)
        encodes it with :class:`PeakEncoder`, summed separately from the
        ``subformula_encoder``'s formula embedding. ``"joint_formula"`` instead
        folds raw intensity into the ``subformula_encoder``'s own projection,
        so one joint layer sees ``[formula_counts, intensity]`` together.
        Requires a ``subformula_encoder`` constructed with ``include_intensity=True``.
    use_global_token : bool, default True
        If ``True``, prepends a CLS token to the peak sequence. It carries:

        - a learned CLS embedding
        - the precursor m/z sinusoidal encoding (if ``use_mz=True``)
        - the metadata embedding (if ``metadata_encoder`` is given)

        If ``False``, no global token is prepended and the transformer sees
        only peak tokens.
    norm_first : bool, default False
        Use a pre-norm transformer stack.
    causal : bool, default False
        If ``True``, applies a causal (lower-triangular) self-attention mask:
        the global/precursor token attends only to itself, and token at position
        ``k`` (``1 <= k <= L``, the ``k``-th peak in caller-supplied order
        attends to positions ``0..k``). Incompatible with ``pool="cls"``.
    """

    def __init__(
        self,
        d_model: int = 512,
        n_layers: int = 8,
        nhead: int = 8,
        dropout: float = 0.15,
        min_mz_wavelength: float = 0.001,
        max_mz_wavelength: float = 10_000,
        pool: str | None = "attention",
        subformula_encoder: nn.Module | None = None,
        metadata_encoder: nn.Module | None = None,
        causal: bool = False,
        use_mz: bool = True,
        intensity_encoding: str = "sinusoidal",
        use_global_token: bool = True,
        norm_first: bool = False,
    ):
        if intensity_encoding not in ("sinusoidal", "joint_formula"):
            raise ValueError(f"Unknown intensity_encoding: {intensity_encoding}")
        super().__init__(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            n_layers=n_layers,
            dropout=dropout,
            peak_encoder=PeakEncoder(
                d_model,
                min_mz_wavelength=min_mz_wavelength,
                max_mz_wavelength=max_mz_wavelength,
                use_mz=use_mz,
                use_intensity=(intensity_encoding == "sinusoidal"),
            ),
        )

        if norm_first:
            layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=d_model * 4,
                dropout=dropout,
                batch_first=True,
                norm_first=True,
            )
            self.transformer_encoder = nn.TransformerEncoder(layer, n_layers)
        self.norm_first = norm_first

        self.precursor_cls = nn.Embedding(1, d_model) if use_global_token else None

        if pool not in ("attention", "cls", "last", None):
            raise ValueError(f"Unknown pool mode: {pool}")
        if causal and pool == "cls":
            raise ValueError("causal=True is incompatible with pool='cls'.")
        self.pool_mode = pool
        self.aggregator = AttnAggregator(d_model) if pool == "attention" else None
        self.causal = causal

        self.subformula_encoder = subformula_encoder

        if metadata_encoder is not None and not use_global_token:
            raise ValueError(
                "metadata_encoder needs use_global_token=True -- metadata is always "
                "added to the global/CLS token"
            )
        if not use_global_token and pool == "cls":
            raise ValueError("pool='cls' needs use_global_token=True")
        if not use_mz and subformula_encoder is None:
            raise ValueError(
                "use_mz=False with no subformula_encoder leaves peaks with no "
                "m/z embedding path at all"
            )
        if intensity_encoding == "joint_formula" and not (
            subformula_encoder is not None and subformula_encoder.include_intensity
        ):
            raise ValueError(
                "intensity_encoding='joint_formula' requires a subformula_encoder "
                "constructed with include_intensity=True"
            )
        self.use_mz = use_mz
        self.intensity_encoding = intensity_encoding
        self.use_global_token = use_global_token
        self.metadata_encoder = metadata_encoder

    def forward(
        self,
        mz: torch.Tensor,
        intensity: torch.Tensor,
        precursor_mz: torch.Tensor,
        subformulae: dict[str, torch.Tensor] | None = None,
        metadata: dict[str, torch.Tensor] | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Encode a batch of spectra into fixed-size embeddings.

        Parameters
        ----------
        mz : torch.Tensor
            ``(B, L)`` float tensor of m/z values (potentially zero-padded).
        intensity : torch.Tensor
            ``(B, L)`` float tensor of intensities (potentially zero-padded).
        precursor_mz : torch.Tensor
            ``(B,)`` float tensor of precursor m/z values.
        subformulae : dict[str, torch.Tensor], optional
            Dict with keys ``"form_vec"``: ``(B, L, ELEMENT_DIM)`` and
            ``"parent_form_vec"``: ``(B, ELEMENT_DIM)``.
            Passed to :class:`SubformulaEncoder` if instantiated, and added to peak embeddings.
        metadata : dict[str, torch.Tensor], optional
            Dict of metadata tensors (see :class:`MetadataEncoder` for the
            accepted keys). Passed to :class:`MetadataEncoder` if instantiated,
            and added to the global/CLS token before the transformer.

        Returns
        -------
        torch.Tensor or tuple[torch.Tensor, torch.Tensor]
            The output depends on the ``pool`` mode set at construction:

            - ``pool in {"attention", "cls", "last"}`` — a ``(B, d_model)``
              float tensor of spectrum embeddings.
            - ``pool is None`` — an ``(out, padding_mask)`` tuple of a
              ``(B, T, d_model)`` tensor and a ``(B, T)`` bool mask with ``True``
              marking padded positions. ``T`` is ``L`` plus one for the global
              token when ``use_global_token`` is set.
        """
        spectra = torch.stack([mz, intensity], dim=2)

        src_key_padding_mask = spectra.sum(dim=2) == 0

        peaks = self.peak_encoder(spectra)
        if self.subformula_encoder is not None and subformulae is not None:
            extra = (
                {"intensity": intensity}
                if self.subformula_encoder.include_intensity
                else {}
            )
            peaks = peaks + self.subformula_encoder(
                subformulae["form_vec"],
                subformulae["parent_form_vec"],
                **extra,
            )

        if self.use_global_token:
            latent_spectra = self.global_token_hook(
                mz_array=mz, intensity_array=intensity, precursor_mzs=precursor_mz
            )
            if self.metadata_encoder is not None and metadata is not None:
                latent_spectra = latent_spectra + self.metadata_encoder(metadata)
            peaks = torch.cat([latent_spectra[:, None, :], peaks], dim=1)
            src_key_padding_mask = torch.cat(
                [
                    src_key_padding_mask.new_zeros(spectra.shape[0], 1),
                    src_key_padding_mask,
                ],
                dim=1,
            )

        if self.causal:
            seq_len = peaks.shape[1]
            attn_mask = torch.triu(
                torch.ones(seq_len, seq_len, dtype=torch.bool, device=peaks.device),
                diagonal=1,
            )
        else:
            attn_mask = None

        out = self.transformer_encoder(
            peaks,
            mask=attn_mask,
            src_key_padding_mask=src_key_padding_mask,
            is_causal=self.causal,
        )

        if self.pool_mode is None:
            return out, src_key_padding_mask
        if self.pool_mode == "cls":
            return out[:, 0, :]
        if self.pool_mode == "last":
            last_idx = (~src_key_padding_mask).sum(dim=1) - 1  # (B,)
            return out[torch.arange(out.shape[0], device=out.device), last_idx]
        return self.aggregator(out, mask=src_key_padding_mask)

    def global_token_hook(
        self,
        mz_array: torch.Tensor,
        intensity_array: torch.Tensor,
        precursor_mzs: torch.Tensor,
    ) -> torch.Tensor:
        """Build the initial global token embedding for a batch of spectra.

        Sums a learned CLS embedding with the sinusoidal encoding of the
        precursor m/z (if ``use_mz=True``).
        Called internally by :meth:`forward` where the
        result is prepended to the peak embeddings.

        Parameters
        ----------
        mz_array : torch.Tensor
            ``(B, L)`` m/z tensor (used only for device context).
        intensity_array : torch.Tensor
            ``(B, L)`` intensity tensor (unused; kept for API compatibility
            with depthcharge's hook signature).
        precursor_mzs : torch.Tensor
            ``(B,)`` precursor m/z tensor.

        Returns
        -------
        torch.Tensor
            ``(B, d_model)`` float tensor.
        """
        precursor_cls_embedding = self.precursor_cls(
            torch.tensor([[0]]).to(mz_array.device)
        )[0].expand(len(mz_array), -1)
        if not self.use_mz:
            return precursor_cls_embedding
        precursor_mz_embedding = self.peak_encoder.mz_encoder(precursor_mzs[:, None])[
            :, 0
        ]
        return precursor_cls_embedding + precursor_mz_embedding


FLARE_ELEMENT_NORM = {
    "H": 102.0,
    "C": 59.0,
    "O": 25.0,
    "N": 13.0,
    "P": 3.0,
    "S": 6.0,
    "Cl": 6.0,
    "F": 17.0,
    "Br": 4.0,
    "I": 4.0,
    "B": 1.0,
    "As": 1.0,
    "Si": 5.0,
    "Se": 2.0,
}


def _resolve_float_norm(float_norm: str | Sequence[float]) -> Sequence[float] | None:

    if float_norm == "flare":
        return [FLARE_ELEMENT_NORM.get(el, 1.0) for el in VALID_ELEMENTS]
    if float_norm == "mistcf":
        return None  # FloatFeaturizer's own default is MIST-CF's common.NORM_VEC.
    if isinstance(float_norm, str):
        raise ValueError(
            f"Unknown float_norm preset: {float_norm!r} "
            "(use 'flare', 'mistcf', or a sequence of floats)"
        )
    return float_norm


class SubformulaEncoder(nn.Module):
    """Encode peak subformulae into ``d_model``-dimensional embeddings.

    Follows the MIST-CF approach: embeds each peak's subformula bag-of-atoms
    and (optionally) its complement (``parent_formula - subformula``),
    concatenates both embeddings, and projects to ``d_model``. The output is
    additively combined with :class:`PeakEncoder` output inside
    :class:`SpectrumEncoder` to make up final peak embeddings.

    The bag-of-atoms vectors consumed by :meth:`forward` (``form_vec`` and
    ``parent_form_vec``) are built from formula strings by
    :func:`~metabo_depthcharge.spec.subformulae.formula_to_dense` (single
    formula) and
    :func:`~metabo_depthcharge.spec.subformulae.assign_peak_subformulae`
    (per-peak subformula assignment across a whole spectrum).

    Parameters
    ----------
    d_model : int
        Output dimension. Must match the transformer's ``d_model``.
    form_embedder : str, default "abs-sines"
        Element-count featurizer used to embed each formula entry, named after
        its MIST-CF implementation. One of:

        - ``"abs-sines"`` — ``FourierFeaturizerAbsoluteSines``
        - ``"fourier"`` — ``FourierFeaturizer``
        - ``"fourier-sines"`` — ``FourierFeaturizerSines``
        - ``"rbf"`` — ``RBFFeaturizer``
        - ``"one-hot"`` — ``OneHotFeaturizer``
        - ``"learnt"`` — ``LearnedFeaturizer``
        - ``"float"`` — ``FloatFeaturizer``: plain per-element normalization
          (``count / norm``), no basis expansion. See ``float_norm``.

        These featurizers are vendored under
        `metabo_depthcharge.mist_cf.nn_utils.form_embedder
        <https://github.com/bittremieuxlab/metabo-depthcharge/blob/main/metabo_depthcharge/mist_cf/nn_utils/form_embedder.py>`_
        from MIST-CF
        (`samgoldman97/mist-cf <https://github.com/samgoldman97/mist-cf>`_).
    use_complement : bool, default True
        If ``False``, the parent-complement half of the embedding is skipped
        (so the input to the projection/MLP head is half as wide).
        Every peak token otherwise carries the precursor formula
        exactly, which is a whole-spectrum constant.
    float_norm : {"flare", "mistcf"} or sequence of float, default "flare"
        Per-element divisor used when ``form_embedder="float"`` (ignored
        otherwise): ``"flare"`` uses FLARE's normalization constants,
        ``"mistcf"`` uses MIST-CF's `common.NORM_VEC``,
        or pass a sequence of floats directly for a custom norm
        (in the same element order as the ``form_vec`` tensors).
    mlp_dims : tuple[int, ...], optional
        If given, the formula embedding is projected to ``d_model`` through an
        MLP with these hidden widths (ReLU + dropout between layers, no final
        activation) instead of a single linear layer. The last entry must
        equal ``d_model``.
    mlp_dropout : float, default 0.2
        Dropout between the ``mlp_dims`` layers. Unused without ``mlp_dims``.
    include_intensity : bool, default False
        If ``True``, ``forward`` takes an additional ``intensity`` tensor and
        concatenates it (one extra raw scalar column) onto the formula
        embedding before the projection/MLP head, so a single joint layer sees
        formula and intensity together. Pair with :class:`SpectrumEncoder`'s
        ``use_intensity=False`` so intensity isn't also separately sinusoidally
        encoded and summed in.
    """

    def __init__(
        self,
        d_model: int,
        form_embedder: str = "abs-sines",
        use_complement: bool = True,
        float_norm: str | Sequence[float] = "flare",
        mlp_dims: tuple[int, ...] | None = None,
        mlp_dropout: float = 0.2,
        include_intensity: bool = False,
    ):
        super().__init__()
        self.form_embedder = form_embedder
        self.use_complement = use_complement
        self.include_intensity = include_intensity
        norm = _resolve_float_norm(float_norm) if form_embedder == "float" else None
        self.form_encoder = get_embedder(form_embedder, norm=norm)

        in_dim = self.form_encoder.full_dim * (2 if use_complement else 1)
        in_dim += 1 if include_intensity else 0
        if mlp_dims is None:
            self.proj = nn.Linear(in_dim, d_model)
            self.layers = None
        else:
            widths = [in_dim, *mlp_dims]
            if widths[-1] != d_model:
                raise ValueError(
                    f"mlp_dims[-1] ({widths[-1]}) must equal d_model ({d_model})"
                )
            self.proj = None
            self.layers = nn.ModuleList(
                nn.Linear(i, o) for i, o in zip(widths, widths[1:], strict=False)
            )
            self.drop = nn.Dropout(mlp_dropout)

    def forward(
        self,
        form_vec: torch.Tensor,
        parent_form_vec: torch.Tensor | None,
        intensity: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Encode per-peak subformulae relative to their parent formula.

        Parameters
        ----------
        form_vec : torch.Tensor
            ``(B, L, ELEMENT_DIM)`` int tensor — bag-of-atoms per peak.
            See :func:`~metabo_depthcharge.spec.subformulae.formula_to_dense` for how to obtain this
            from a formula string.
        parent_form_vec : torch.Tensor, optional
            ``(B, ELEMENT_DIM)`` int tensor — parent molecular formula.
            See :func:`~metabo_depthcharge.spec.subformulae.formula_to_dense` for how to obtain this
            from a formula string. Unused when ``use_complement=False``.
        intensity : torch.Tensor, optional
            ``(B, L)`` float tensor of raw peak intensities. Required when
            ``include_intensity=True``, ignored otherwise.

        Returns
        -------
        torch.Tensor
            ``(B, L, d_model)`` float tensor to be added to peak embeddings.
        """
        form_emb = self.form_encoder(form_vec)  # (B, L, full_dim)
        if self.use_complement:
            diff_vec = parent_form_vec[:, None, :] - form_vec  # (B, L, ELEMENT_DIM)
            diff_emb = self.form_encoder(diff_vec)  # (B, L, full_dim)
            x = torch.cat([form_emb, diff_emb], dim=-1)
        else:
            x = form_emb

        if self.include_intensity:
            if intensity is None:
                raise ValueError("include_intensity=True requires an intensity tensor")
            x = torch.cat([x, intensity[..., None].to(x.dtype)], dim=-1)

        if self.proj is not None:
            return self.proj(x)
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:
                x = self.drop(functional.relu(x))
        return x
