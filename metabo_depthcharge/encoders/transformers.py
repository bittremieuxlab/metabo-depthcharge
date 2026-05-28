import torch
import torch.nn as nn
from depthcharge.encoders import FloatEncoder
from depthcharge.transformers import SpectrumTransformerEncoder

from metabo_depthcharge.spec.adducts import N_ADDUCTS
from metabo_depthcharge.spec.metadata_parsers import N_INSTRUMENTS


class MetadataEncoder(nn.Module):
    """Encodes spectrum acquisition metadata into a d_model vector.

    Each enabled field gets its own encoder. Outputs are summed.
    Missing values (index 0 for categoricals, 0.0 for CE) contribute
    zero to the output for that sample.

    Args:
        d_model: Output embedding dimension.
        metadata_fields: List of field names to encode
            (subset of ["adduct", "collision_energy", "instrument_type"]).
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
        """Encode metadata into a (B, d_model) vector.

        Args:
            metadata: Dict mapping field names to tensors.
                - "adduct": (B,) int64 indices
                - "collision_energy": (B,) float32 values
                - "instrument_type": (B,) int64 indices

        Returns:
            (B, d_model) tensor — sum of all enabled field embeddings.
        """
        parts = []

        if "adduct" in self.metadata_fields and "adduct" in metadata:
            parts.append(self.adduct_emb(metadata["adduct"].long()))

        if (
            "collision_energy" in self.metadata_fields
            and "collision_energy" in metadata
        ):
            ce = metadata["collision_energy"].float()
            ce_emb = self.ce_encoder(ce[:, None])  # (B, d_model)
            # Zero out embedding for missing CE (value == 0)
            mask = (ce != 0.0).unsqueeze(-1)  # (B, 1)
            parts.append(ce_emb * mask)

        if "instrument_type" in self.metadata_fields and "instrument_type" in metadata:
            parts.append(self.instrument_emb(metadata["instrument_type"].long()))

        if not parts:
            # Return zeros — get device/dtype from any metadata tensor
            any_tensor = next(iter(metadata.values()))
            return torch.zeros(
                any_tensor.shape[0],
                self.adduct_emb.embedding_dim
                if hasattr(self, "adduct_emb")
                else self.instrument_emb.embedding_dim,
                device=any_tensor.device,
                dtype=any_tensor.dtype,
            )

        return sum(parts)


class PeakEncoder(nn.Module):
    def __init__(self, d_model, min_mz_wavelength, max_mz_wavelength):
        """Cheeky customized PeakEncoder that does not concatenate+project
        mz encoding and int encoding but sums them instead.
        """
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

    def forward(self, x):
        return self.mz_encoder(x[:, :, 0]) + self.int_encoder(x[:, :, 1])


class AttnAggregator(nn.Module):
    def __init__(
        self,
        hidden_dim: int = None,
    ):
        super().__init__()

        self.to_attn_logits = nn.Linear(hidden_dim, 1)

    def forward(self, x, mask=None):  # ..., L, D
        """
        Forward pass with optional masking. Aggregates over the second-to-last
        dimension, so works with any number of leading batch dimensions.

        Args:
            x: Input tensor of shape (..., L, D)
            mask: Optional boolean mask of shape (..., L) where False indicates
                  positions to keep and True indicates positions to mask out.

        Returns:
            Aggregated tensor of shape (..., D)
        """
        attn_logits = self.to_attn_logits(x)  # ..., L, 1

        if mask is not None:
            attn_logits = attn_logits.masked_fill(mask.unsqueeze(-1), float("-inf"))

        attn_values = attn_logits.softmax(dim=-2)  # ..., L, 1

        return (x * attn_values).sum(-2)  # ..., D


class ResidualProjection(nn.Module):
    """Residual projection mapper.

    Architecture:
    - Initial linear layer: d_in -> d_out (skipped if d_in == d_out and n_layers == 0)
    - N residual blocks: LayerNorm -> inverted bottleneck MLP (d_out -> 4*d_out -> d_out)

    When n_layers=0: just Linear(d_in, d_out) + LayerNorm (or Identity if d_in == d_out).
    When n_layers>0: Linear(d_in, d_out) + N residual blocks.
    """

    def __init__(self, d_in, d_out, n_layers=0, dropout=0.10):
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

    def forward(self, x):
        x = self.init_proj(x)
        if self.blocks is not None:
            for block in self.blocks:
                x = x + block(x)
        return self.norm(x)


class DepthchargeEncoder(SpectrumTransformerEncoder):
    """Spectrum encoder based on depthcharge's SpectrumTransformerEncoder.

    Wraps SpectrumTransformerEncoder + pooling + projection into a single
    module with the contract: forward(mz, intensity, precursor_mz) -> (B, d_out).

    Args:
        d_model: Transformer hidden dimension.
        n_layers: Number of transformer layers.
        dropout: Dropout rate.
        min_mz_wavelength: Min wavelength for m/z positional encoding.
        max_mz_wavelength: Max wavelength for m/z positional encoding.
        pool: Pooling method - 'attention' or 'cls'.
        d_out: Output embedding dimension (after projection).
        n_proj_layers: Number of residual projection blocks (0 = linear only).
    """

    def __init__(
        self,
        d_model=512,
        n_layers=8,
        dropout=0.15,
        min_mz_wavelength=0.001,
        max_mz_wavelength=10_000,
        pool="attention",
        d_out=512,
        n_proj_layers=0,
        subformula_encoder=None,
        metadata_encoder=None,
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
        mz,
        intensity,
        precursor_mz,
        form_vec=None,
        parent_form_vec=None,
        metadata=None,
    ):
        spectra = torch.stack([mz, intensity], dim=2)

        src_key_padding_mask = spectra.sum(dim=2) == 0
        global_token_mask = torch.tensor([[False]] * spectra.shape[0]).type_as(
            src_key_padding_mask
        )
        src_key_padding_mask = torch.cat(
            [global_token_mask, src_key_padding_mask], dim=1
        )

        peaks = self.peak_encoder(spectra)

        if self.subformula_encoder is not None and form_vec is not None:
            peaks = peaks + self.subformula_encoder(form_vec, parent_form_vec)

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
    ):
        precursor_cls_embedding = self.precursor_cls(
            torch.tensor([[0]]).to(mz_array.device)
        )[0].expand(len(mz_array), -1)
        precursor_mz_embedding = self.peak_encoder.mz_encoder(precursor_mzs[:, None])[
            :, 0
        ]
        return precursor_cls_embedding + precursor_mz_embedding
