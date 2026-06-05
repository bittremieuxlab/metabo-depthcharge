"""Encoders: neural encoders for spectra, molecules, metadata, subformulae."""

from metabo_depthcharge.encoders.molecules import MolMLP, MultiMolMLP
from metabo_depthcharge.encoders.nn import AttnAggregator, ResidualProjection
from metabo_depthcharge.encoders.spectra import (
    MetadataEncoder,
    PeakEncoder,
    SpectrumEmbedder,
    SubformulaEncoder,
)


__all__ = [
    "AttnAggregator",
    "SpectrumEmbedder",
    "MetadataEncoder",
    "MolMLP",
    "MultiMolMLP",
    "PeakEncoder",
    "ResidualProjection",
    "SubformulaEncoder",
]
