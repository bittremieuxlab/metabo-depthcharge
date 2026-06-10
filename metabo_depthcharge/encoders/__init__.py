"""Encoders: neural encoders for spectra, molecules, metadata, subformulae."""

from metabo_depthcharge.encoders.molecules import MolMLP, MultiMolMLP
from metabo_depthcharge.encoders.nn import AttnAggregator, ResidualNetwork
from metabo_depthcharge.encoders.spectra import (
    MetadataEncoder,
    PeakEncoder,
    SpectrumEncoder,
    SubformulaEncoder,
)


__all__ = [
    "AttnAggregator",
    "SpectrumEncoder",
    "MetadataEncoder",
    "MolMLP",
    "MultiMolMLP",
    "PeakEncoder",
    "ResidualNetwork",
    "SubformulaEncoder",
]
