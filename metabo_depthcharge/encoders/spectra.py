import warnings

import torch
import torch.nn as nn
from depthcharge.encoders import FloatEncoder
from depthcharge.transformers import SpectrumTransformerEncoder

from metabo_depthcharge.encoders.nn import AttnAggregator
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
            # Index 0 = unknown/other, padding_idx=0 so unknown → zero vector
            self.adduct_emb = nn.Embedding(N_ADDUCTS, d_model, padding_idx=0)

        if "collision_energy" in metadata_fields:
            self.ce_encoder = nn.Linear(1, d_model)

        if "instrument_type" in metadata_fields:
            self.instrument_emb = nn.Embedding(N_INSTRUMENTS, d_model, padding_idx=0)

        if "ion_activation" in metadata_fields:
            self.ion_activation_emb = nn.Embedding(
                N_ION_ACTIVATIONS, d_model, padding_idx=0
            )

        if "ionization_method" in metadata_fields:
            self.ionization_method_emb = nn.Embedding(
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

    References
    ----------
    .. [1] Vaswani, Ashish, et al. "Attention Is All You
       Need." Advances in Neural Information Processing Systems 30 (2017).
       https://arxiv.org/abs/1706.03762
    """

    def __init__(
        self, d_model: int, min_mz_wavelength: float, max_mz_wavelength: float
    ):
        super().__init__()
        self.d_model = d_model

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
        return self.mz_encoder(x[:, :, 0]) + self.int_encoder(x[:, :, 1])


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
        (global/first token), or ``None`` to skip pooling and return the full
        token sequence together with its padding mask.
    subformula_encoder : nn.Module, optional
        :class:`SubformulaEncoder` whose output is added to peak embeddings
        before the transformer.
    metadata_encoder : nn.Module, optional
        :class:`MetadataEncoder` whose output is added to the global/CLS
        token before the transformer.
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
    ):
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
            ),
        )

        self.precursor_cls = nn.Embedding(1, d_model)

        if pool not in ("attention", "cls", None):
            raise ValueError(f"Unknown pool mode: {pool}")
        self.pool_mode = pool
        self.aggregator = AttnAggregator(d_model) if pool == "attention" else None

        self.subformula_encoder = subformula_encoder
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
            and its ``(B, d_model)`` output is added to the global/CLS token
            (prepended at index 0, alongside the learned CLS embedding and the
            precursor-m/z encoding) before the transformer.

        Returns
        -------
        torch.Tensor or tuple[torch.Tensor, torch.Tensor]
            The output depends on the ``pool`` mode set at construction:

            - ``pool in {"attention", "cls"}`` — a ``(B, d_model)`` float tensor
              of spectrum embeddings.
            - ``pool is None`` — an ``(out, padding_mask)`` tuple, where ``out`` is a
              ``(B, L + 1, d_model)`` tensor (global token at index 0 followed
              by the peak tokens) and ``padding_mask`` is a ``(B, L + 1)`` bool
              tensor with ``True`` marking padded positions.
        """
        spectra = torch.stack([mz, intensity], dim=2)

        src_key_padding_mask = spectra.sum(dim=2) == 0
        global_token_mask = torch.tensor([[False]] * spectra.shape[0]).type_as(
            src_key_padding_mask
        )
        src_key_padding_mask = torch.cat(
            [global_token_mask, src_key_padding_mask], dim=1
        )

        peaks = self.peak_encoder(spectra)

        if self.subformula_encoder is not None and subformulae is not None:
            peaks = peaks + self.subformula_encoder(
                subformulae["form_vec"], subformulae["parent_form_vec"]
            )

        latent_spectra = self.global_token_hook(
            mz_array=mz, intensity_array=intensity, precursor_mzs=precursor_mz
        )

        # Add metadata embedding to the global/CLS token
        if self.metadata_encoder is not None and metadata is not None:
            latent_spectra = latent_spectra + self.metadata_encoder(metadata)

        peaks = torch.cat([latent_spectra[:, None, :], peaks], dim=1)

        out = self.transformer_encoder(
            peaks, mask=None, src_key_padding_mask=src_key_padding_mask
        )

        if self.pool_mode is None:
            return out, src_key_padding_mask
        if self.pool_mode == "cls":
            return out[:, 0, :]
        return self.aggregator(out, mask=src_key_padding_mask)

    def global_token_hook(
        self,
        mz_array: torch.Tensor,
        intensity_array: torch.Tensor,
        precursor_mzs: torch.Tensor,
    ) -> torch.Tensor:
        """Build the initial global token embedding for a batch of spectra.

        Sums a learned CLS embedding with the sinusoidal encoding of the
        precursor m/z. Called internally by :meth:`forward` where the result
        is prepended to the peak embeddings.

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
        precursor_mz_embedding = self.peak_encoder.mz_encoder(precursor_mzs[:, None])[
            :, 0
        ]
        return precursor_cls_embedding + precursor_mz_embedding


class SubformulaEncoder(nn.Module):
    """Encode peak subformulae into ``d_model``-dimensional embeddings.

    Follows the MIST-CF approach: embeds each peak's subformula bag-of-atoms
    and its complement (``parent_formula - subformula``), concatenates both
    embeddings, and linearly projects to ``d_model``. The output is additively combined with
    :class:`PeakEncoder` output inside :class:`SpectrumEncoder` to make up final
    peak embeddings.

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
        - ``"float"`` — ``FloatFeaturizer``

        These featurizers are vendored under
        `metabo_depthcharge.mist_cf.nn_utils.form_embedder
        <https://github.com/bittremieuxlab/metabo-depthcharge/blob/main/metabo_depthcharge/mist_cf/nn_utils/form_embedder.py>`_
        from MIST-CF
        (`samgoldman97/mist-cf <https://github.com/samgoldman97/mist-cf>`_).
    """

    def __init__(self, d_model: int, form_embedder: str = "abs-sines"):
        super().__init__()
        self.form_encoder = get_embedder(form_embedder)
        self.proj = nn.Linear(self.form_encoder.full_dim * 2, d_model)

    def forward(
        self,
        form_vec: torch.Tensor,
        parent_form_vec: torch.Tensor,
    ) -> torch.Tensor:
        """Encode per-peak subformulae relative to their parent formula.

        Parameters
        ----------
        form_vec : torch.Tensor
            ``(B, L, ELEMENT_DIM)`` int tensor — bag-of-atoms per peak.
            See :func:`~metabo_depthcharge.spec.subformulae.formula_to_dense` for how to obtain this
            from a formula string.
        parent_form_vec : torch.Tensor
            ``(B, ELEMENT_DIM)`` int tensor — parent molecular formula.
            See :func:`~metabo_depthcharge.spec.subformulae.formula_to_dense` for how to obtain this
            from a formula string.

        Returns
        -------
        torch.Tensor
            ``(B, L, d_model)`` float tensor to be added to peak embeddings.
        """
        diff_vec = parent_form_vec[:, None, :] - form_vec  # (B, L, ELEMENT_DIM)
        form_emb = self.form_encoder(form_vec)  # (B, L, full_dim)
        diff_emb = self.form_encoder(diff_vec)  # (B, L, full_dim)
        return self.proj(torch.cat([form_emb, diff_emb], dim=-1))  # (B, L, d_model)
