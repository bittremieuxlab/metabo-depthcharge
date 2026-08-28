Chem
====

``metabo_depthcharge.chem`` provides per-molecule primitives: a canonicalizing
``Molecule`` wrapper, fingerprint/embedding/graph representation generators, and
molecular similarity measures.

.. currentmodule:: metabo_depthcharge.chem

Molecule
--------

.. autosummary::
   :toctree: generated

   Molecule

Representations
---------------

.. autosummary::
   :toctree: generated

   MoleculeToMorgan
   MoleculeToRdkit
   MoleculeToMACCS
   MoleculeToBiosynfoni
   MoleculeToMAP4
   MoleculeToMolFormer
   MoleculeToChemBERTa
   MoleculeToSAFEGPT
   MoleculeToGraph

Similarities
------------

.. autosummary::
   :toctree: generated

   BinaryTanimoto
   CountTanimoto
   CosineSimilarity
   MCESDistance
