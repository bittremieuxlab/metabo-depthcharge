import torch
import torch.nn as nn
from depthcharge.encoders import FloatEncoder
from depthcharge.transformers import SpectrumTransformerEncoder

from metabo_depthcharge.encoders.nn import AttnAggregator, ResidualProjection
from metabo_depthcharge.mist_cf.nn_utils import get_embedder
from metabo_depthcharge.spec.adducts import N_ADDUCTS
from metabo_depthcharge.spec.metadata_parsers import N_INSTRUMENTS


class MetadataEncoder(nn.Module):
    """Encode spectrum acquisition metadata into a ``d_model`` vector.

    Each enabled field gets its own encoder. Outputs are summed.
    Missing values (index 0 for categoricals, 0.0 for CE) contribute
    zero to the output for that sample.

    Parameters
    ----------
    d_model : int
        Output embedding dimension.
    metadata_fields : list[str]
        Field names to encode; subset of
        ``["adduct", "collision_energy", "instrument_type"]``.
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

        if not parts:
            return 0

        return sum(parts)


class PeakEncoder(nn.Module):
    """Encode (m/z, intensity) peak pairs into ``d_model``-dimensional vectors.

    Differs from depthcharge's default ``PeakEncoder``: instead of concatenating
    the m/z and intensity encodings and projecting, it sums them, matching
    Vaswani et al. original transformer formulation.

    Parameters
    ----------
    d_model : int
        Output embedding dimension.
    min_mz_wavelength : float
        Minimum wavelength for m/z sinusoidal encoding.
    max_mz_wavelength : float
        Maximum wavelength for m/z sinusoidal encoding.
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


class SpectrumEmbedder(SpectrumTransformerEncoder):
    """Spectrum encoder based on depthcharge's ``SpectrumTransformerEncoder``.

    Wraps ``SpectrumTransformerEncoder`` + pooling + projection into a single
    module with the contract:
    ``forward(mz, intensity, precursor_mz) -> (B, d_out)``.

    Parameters
    ----------
    d_model : int, default 512
        Transformer hidden dimension.
    n_layers : int, default 8
        Number of transformer layers.
    dropout : float, default 0.15
        Dropout rate.
    min_mz_wavelength : float, default 0.001
        Min wavelength for m/z positional encoding.
    max_mz_wavelength : float, default 10000
        Max wavelength for m/z positional encoding.
    pool : str, default "attention"
        Pooling method — ``"attention"`` (learned weighted sum) or
        ``"cls"`` (first token).
    d_out : int, default 512
        Output embedding dimension (after projection).
    n_proj_layers : int, default 0
        Number of residual blocks in the output projection
        (0 = linear only).
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
        dropout: float = 0.15,
        min_mz_wavelength: float = 0.001,
        max_mz_wavelength: float = 10_000,
        pool: str = "attention",
        d_out: int = 512,
        n_proj_layers: int = 0,
        subformula_encoder: nn.Module | None = None,
        metadata_encoder: nn.Module | None = None,
    ):
        super().__init__(
            d_model=d_model,
            nhead=8,
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

        self.pool_mode = pool
        if pool == "attention":
            self.aggregator = AttnAggregator(d_model)
        elif pool == "cls":
            self.aggregator = None
        else:
            raise ValueError(f"Unknown pool mode: {pool}")

        self.subformula_encoder = subformula_encoder
        self.metadata_encoder = metadata_encoder

        self.proj = ResidualProjection(
            d_model, d_out, n_layers=n_proj_layers, dropout=dropout
        )

    def forward(
        self,
        mz: torch.Tensor,
        intensity: torch.Tensor,
        precursor_mz: torch.Tensor,
        subformulae: dict[str, torch.Tensor] | None = None,
        metadata: dict[str, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        """Encode a batch of spectra into fixed-size embeddings.

        Parameters
        ----------
        mz : torch.Tensor
            ``(B, L)`` float tensor of m/z values (zero-padded).
        intensity : torch.Tensor
            ``(B, L)`` float tensor of intensities (zero-padded).
        precursor_mz : torch.Tensor
            ``(B,)`` float tensor of precursor m/z values.
        subformulae : dict[str, torch.Tensor], optional
            Dict with keys ``"form_vec"`` ``(B, L, ELEMENT_DIM)`` and
            ``"parent_form_vec"`` ``(B, ELEMENT_DIM)`` for
            subformula-augmented encoding.
        metadata : dict[str, torch.Tensor], optional
            Metadata tensors passed to :class:`MetadataEncoder`.

        Returns
        -------
        torch.Tensor
            ``(B, d_out)`` float tensor of spectrum embeddings.
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

        if self.pool_mode == "cls":
            pooled = out[:, 0, :]
        else:
            pooled = self.aggregator(out, mask=src_key_padding_mask)

        return self.proj(pooled)

    def global_token_hook(
        self,
        mz_array: torch.Tensor,
        intensity_array: torch.Tensor,
        precursor_mzs: torch.Tensor,
    ) -> torch.Tensor:
        """Build the initial global token embedding for a batch of spectra.

        Sums a learned CLS embedding with the sinusoidal encoding of the
        precursor m/z. Called internally by :meth:`forward`.

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

    Follows the MIST approach: embeds each peak's subformula bag-of-atoms
    and its complement (``parent_formula - subformula``), concatenates both
    embeddings, and projects to ``d_model``. The output is intended to be
    added to :class:`PeakEncoder` output (additive fusion) inside
    :class:`SpectrumEmbedder`.

    Parameters
    ----------
    d_model : int
        Output dimension; must match the transformer's ``d_model``.
    form_embedder : str, default "abs-sines"
        MIST formula embedder type. One of ``"abs-sines"``, ``"fourier"``,
        ``"fourier-sines"``, ``"rbf"``, ``"one-hot"``, ``"learnt"``,
        ``"float"``.
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
        parent_form_vec : torch.Tensor
            ``(B, ELEMENT_DIM)`` int tensor — parent molecular formula.

        Returns
        -------
        torch.Tensor
            ``(B, L, d_model)`` float tensor to be added to peak embeddings.
        """
        diff_vec = parent_form_vec[:, None, :] - form_vec  # (B, L, ELEMENT_DIM)
        form_emb = self.form_encoder(form_vec)  # (B, L, full_dim)
        diff_emb = self.form_encoder(diff_vec)  # (B, L, full_dim)
        return self.proj(torch.cat([form_emb, diff_emb], dim=-1))  # (B, L, d_model)
