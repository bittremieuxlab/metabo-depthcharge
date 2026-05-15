"""Molecule class"""

import functools
import warnings
from functools import cached_property

from rdkit import Chem
from rdkit.Chem import Descriptors, SanitizeFlags, rdMolDescriptors
from rdkit.Chem.MolStandardize import rdMolStandardize


PROPERTIES: tuple[str, ...] = (
    "canonical_smiles",
    "inchi",
    "inchikey",
    "inchikey_2d",
    "formula",
    "exact_mass",
)


# Cached so under ``Dataset.map(num_proc=N)`` each worker constructs these
# once on first use, not per row.
@functools.cache
def _uncharger() -> rdMolStandardize.Uncharger:
    return rdMolStandardize.Uncharger()


@functools.cache
def _tautomerizer() -> rdMolStandardize.TautomerEnumerator:
    return rdMolStandardize.TautomerEnumerator()


def _lenient_mol_from_smiles(smiles: str) -> Chem.Mol:
    """Parse a SMILES with progressive sanitization fallbacks.

    Tries strict parsing first, then drops ``SANITIZE_PROPERTIES`` (valence
    checks), then also drops ``SANITIZE_KEKULIZE``, and finally returns the
    unsanitized mol if all else fails. Raises only if the SMILES can't be
    parsed at all (even without sanitization).
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is not None:
        return mol

    mol = Chem.MolFromSmiles(smiles, sanitize=False)
    if mol is None:
        raise ValueError(f"Cannot parse SMILES even without sanitization: {smiles!r}")

    for skip in (
        SanitizeFlags.SANITIZE_PROPERTIES,
        SanitizeFlags.SANITIZE_PROPERTIES | SanitizeFlags.SANITIZE_KEKULIZE,
    ):
        try:
            Chem.SanitizeMol(mol, sanitizeOps=SanitizeFlags.SANITIZE_ALL ^ skip)
            warnings.warn(
                f"Fallback parsing (skip {skip!r}) used for SMILES: {smiles!r}",
                stacklevel=3,
            )
            return mol
        except Exception:
            continue

    warnings.warn(
        f"Fallback parsing (UNSANITIZED) used for SMILES: {smiles!r}", stacklevel=3
    )
    return mol


class Molecule:
    """Cached wrapper around an RDKit :class:`Mol`.

    Parameters
    ----------
    smiles : str
        SMILES string identifying the molecule.
    mol : rdkit.Chem.Mol, optional, keyword-only
        Pre-built RDKit Mol; supplying it skips the SMILES re-parse on
        :attr:`mol` access.
    """

    def __init__(
        self,
        smiles: str,
        mol: Chem.Mol | None = None,
    ):
        self.smiles = smiles
        if mol is not None:
            self.__dict__["mol"] = mol

    def __repr__(self) -> str:
        return f"Molecule(smiles={self.smiles!r})"

    @classmethod
    def from_dict(cls, row: dict) -> "Molecule":
        """Construct from a row-like dict, pre-populating the cache.

        Any keys in ``row`` that match a name in :data:`PROPERTIES` are
        copied into the instance so accessing those properties returns
        the stored value without invoking RDKit. The ``smiles`` key is
        required.

        Parameters
        ----------
        row : dict
            Row-like mapping (e.g. a :class:`MoleculeDataset` item, a
            pandas row). Must contain a ``smiles`` key.
        """
        m = cls(smiles=row["smiles"])
        for prop in PROPERTIES:
            if prop in row:
                m.__dict__[prop] = row[prop]
        return m

    @cached_property
    def mol(self) -> Chem.Mol:
        return _lenient_mol_from_smiles(self.smiles)

    @cached_property
    def canonical_smiles(self) -> str:
        return Chem.MolToSmiles(self.mol)

    @cached_property
    def inchi(self) -> str:
        return Chem.MolToInchi(self.mol)

    @cached_property
    def inchikey(self) -> str:
        return Chem.MolToInchiKey(self.mol)

    @cached_property
    def inchikey_2d(self) -> str:
        return self.inchikey.split("-")[0]

    @cached_property
    def formula(self) -> str:
        return rdMolDescriptors.CalcMolFormula(self.mol)

    @cached_property
    def exact_mass(self) -> float:
        return Descriptors.ExactMolWt(self.mol)

    def standardize(
        self,
        *,
        remove_stereo: bool = True,
        uncharge: bool = True,
        canonicalize_tautomers: bool = False,
    ) -> "Molecule | None":
        """Return a standardized copy, or ``None`` if RDKit can't clean it.

        Always runs ``Cleanup`` (sanitize / normalize / reionize) and
        ``FragmentParent`` (drop counterions); other steps are opt-in.
        Returns ``None`` when standardization throws.

        Parameters
        ----------
        remove_stereo : bool, default True
            Drop stereochemistry before standardizing.
        uncharge : bool, default True
            Neutralize charges via RDKit's ``Uncharger``.
        canonicalize_tautomers : bool, default False
            Run ``TautomerEnumerator.Canonicalize``.

        Returns
        -------
        Molecule or None
            New :class:`Molecule` with standardized SMILES on success;
            ``None`` on failure.
        """
        try:
            if remove_stereo:
                Chem.RemoveStereochemistry(self.mol)
            m = rdMolStandardize.Cleanup(self.mol)
            m = rdMolStandardize.FragmentParent(m)
            if uncharge:
                m = _uncharger().uncharge(m)
            if canonicalize_tautomers:
                m = _tautomerizer().Canonicalize(m)
        except Exception:
            return None
        return Molecule(Chem.MolToSmiles(m), mol=m)
