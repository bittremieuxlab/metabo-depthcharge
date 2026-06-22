Spec
====

``metabo_depthcharge.spec`` provides per-spectrum primitives: spectrum
preprocessing, adduct/metadata encoding, and MIST-style peak subformula
assignment.

Spectrum preprocessing
----------------------

.. currentmodule:: metabo_depthcharge.spec.preprocessing

.. autosummary::
   :toctree: generated

   Spectrum
   Normalizer
   Trimmer
   PeakFilter
   CollapseSteppedCE
   SequentialPreprocessor
   DefaultSpectrumProcessor

Adducts
-------

.. currentmodule:: metabo_depthcharge.spec.adducts

.. autosummary::
   :toctree: generated

   ADDUCT_VOCAB
   ADDUCT_MASS
   encode_adduct
   mz_to_neutral_mass
   neutral_mass_to_mz

Metadata parsing
----------------

.. currentmodule:: metabo_depthcharge.spec.metadata_parsers

.. autosummary::
   :toctree: generated

   encode_instrument
   encode_collision_energy
   encode_ion_activation
   encode_ionization_method
   INSTRUMENT_TYPES
   ION_ACTIVATION_METHODS
   IONIZATION_METHODS
   METADATA_FIELDS
   METADATA_PARSERS

Subformulae
-----------

.. currentmodule:: metabo_depthcharge.spec.subformulae

.. autosummary::
   :toctree: generated

   formula_to_dense
   assign_peak_subformulae
