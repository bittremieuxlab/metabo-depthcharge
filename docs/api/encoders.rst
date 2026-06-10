Encoders
========

``metabo_depthcharge.encoders`` provides neural network modules for encoding
spectra, molecules, metadata, and subformulae.

.. currentmodule:: metabo_depthcharge.encoders

Spectrum encoders
-----------------

.. autosummary::
   :toctree: generated

   SpectrumEncoder
   PeakEncoder
   MetadataEncoder
   SubformulaEncoder

Molecule encoders
-----------------

.. autosummary::
   :toctree: generated

   MolMLP
   MultiMolMLP

Shared building blocks
----------------------

.. autosummary::
   :toctree: generated

   AttnAggregator
   ResidualNetwork
