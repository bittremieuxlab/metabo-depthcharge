# Quickstart

These are currently AI-generated stubs / place-holders.
The [API reference](api/index.rst) should be better for the moment now.

:::{rubric} Molecules and fingerprints
:::

```python
from metabo_depthcharge.chem import Molecule, MoleculeToMorgan

mol = Molecule("CC(=O)Oc1ccccc1C(=O)O")  # aspirin
fp = MoleculeToMorgan(radius=2, n_bits=2048)(mol)
```

:::{rubric} Spectrum preprocessing
:::

```python
from metabo_depthcharge.spec import Spectrum, DefaultSpectrumProcessor

spec = Spectrum(mz=[...], intensity=[...], precursor_mz=195.1)
processed = DefaultSpectrumProcessor(spec)
```

:::{rubric} Datasets
:::

```python
from metabo_depthcharge.datasets import SpectrumDataset, MoleculeDataset

# Spectra from an MGF file
spectra = SpectrumDataset.from_mgf("path/to/spectra.mgf")
item = spectra[0]  # torch tensors

# Molecules from a TSV with a SMILES column
mols = MoleculeDataset.from_tsv("path/to/molecules.tsv", smiles_column="smiles")
```

:::{rubric} Similarities
:::

```python
from metabo_depthcharge.chem import BinaryTanimoto, MCESDistance

sim = BinaryTanimoto()
d = MCESDistance()
```
