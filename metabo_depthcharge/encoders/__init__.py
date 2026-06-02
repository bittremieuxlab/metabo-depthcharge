"""Encoders: neural encoders for spectra, molecules, metadata, subformulae."""

from metabo_depthcharge.encoders.molecules import MolEmbedder, MultiMolEmbedder
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
    "MolEmbedder",
    "MultiMolEmbedder",
    "PeakEncoder",
    "ResidualProjection",
    "SubformulaEncoder",
]
