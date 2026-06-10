"""Molecule class"""

import functools
import warnings
from functools import cached_property

from rdkit import Chem, rdBase
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
        Pre-built RDKit Mol. Supplying it skips the SMILES re-parse on
        :attr:`mol` access.

    Notes
    -----
    Derived attributes (:attr:`mol`, :attr:`canonical_smiles`,
    :attr:`inchikey`, ...) are computed lazily on first access and cached
    for the lifetime of the instance. Treat instances as immutable:
    reassigning :attr:`smiles` does **not** refresh already-cached values, so
    build a new :class:`Molecule` rather than mutating one.

    Parsing is lenient. Accessing :attr:`mol` (and anything derived from it)
    never hard-fails on a slightly malformed SMILES -- it retries with
    progressively relaxed sanitization and emits a :func:`warnings.warn`,
    raising only when the string cannot be parsed at all. :attr:`smiles` is
    stored verbatim. :attr:`canonical_smiles` is the RDKit-canonical form.

    Equality and hashing use :attr:`inchikey_2d` (the 2D connectivity layer),
    so stereoisomers and charge/protonation variants compare equal and
    collide as ``set`` / ``dict`` keys. Use :attr:`inchikey` directly for
    stereo-sensitive identity. Hashing computes the InChIKey (an RDKit parse)
    on first use.
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

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Molecule):
            return NotImplemented
        return self.inchikey_2d == other.inchikey_2d

    def __hash__(self) -> int:
        return hash(self.inchikey_2d)

    @classmethod
    def from_dict(cls, row: dict) -> "Molecule":
        """Construct from a row-like dict, pre-populating the cache.

        Any keys in ``row`` that match one of the class attribute names are
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

    def standardize(
        self,
        *,
        remove_stereo: bool = True,
        uncharge: bool = False,
        canonicalize_tautomers: bool = False,
    ) -> "Molecule | None":
        """Return a standardized copy, or ``None`` if RDKit can't clean it.

        Always runs :func:`~rdkit.Chem.MolStandardize.rdMolStandardize.Cleanup`
        (sanitize / normalize / reionize) and
        :func:`~rdkit.Chem.MolStandardize.rdMolStandardize.FragmentParent`
        (drop counterions); other steps are opt-in. This instance is never
        modified -- standardization runs on a copy and the result is returned
        as a new :class:`Molecule`.

        Parameters
        ----------
        remove_stereo : bool, default True
            Drop stereochemistry before standardizing.
        uncharge : bool, default False
            Neutralize charges via RDKit's
            :class:`~rdkit.Chem.MolStandardize.rdMolStandardize.Uncharger`.
        canonicalize_tautomers : bool, default False
            Run :meth:`TautomerEnumerator.Canonicalize
            <rdkit.Chem.MolStandardize.rdMolStandardize.TautomerEnumerator.Canonicalize>`.

        Returns
        -------
        Molecule or None
            A new standardized :class:`Molecule` on success; ``None`` on
            failure.
        """
        mol = Chem.Mol(self.mol)  # copy; never mutate self
        try:
            if remove_stereo:
                Chem.RemoveStereochemistry(mol)
            m = rdMolStandardize.Cleanup(mol)
            m = rdMolStandardize.FragmentParent(m)
            if uncharge:
                m = _uncharger().uncharge(m)
            if canonicalize_tautomers:
                m = _tautomerizer().Canonicalize(m)
            out_smi = Chem.MolToSmiles(m)
        except Exception:
            return None
        with rdBase.BlockLogs():
            if Chem.MolFromSmiles(out_smi) is None:
                return None
        return Molecule(out_smi, mol=m)

    @cached_property
    def mol(self) -> Chem.Mol:
        """RDKit :class:`~rdkit.Chem.Mol` parsed from :attr:`smiles`."""
        return _lenient_mol_from_smiles(self.smiles)

    @cached_property
    def canonical_smiles(self) -> str:
        """RDKit canonical SMILES string."""
        return Chem.MolToSmiles(self.mol)

    @cached_property
    def inchi(self) -> str:
        """Standard InChI string."""
        return Chem.MolToInchi(self.mol)

    @cached_property
    def inchikey(self) -> str:
        """Standard InChIKey (27-character hash of the InChI)."""
        return Chem.MolToInchiKey(self.mol)

    @cached_property
    def inchikey_2d(self) -> str:
        """First block of the InChIKey (skeleton/connectivity, no stereo or charge)."""
        return self.inchikey.split("-")[0]

    @cached_property
    def formula(self) -> str:
        """Molecular formula in Hill notation."""
        return rdMolDescriptors.CalcMolFormula(self.mol)

    @cached_property
    def exact_mass(self) -> float:
        """Monoisotopic exact mass in daltons."""
        return Descriptors.ExactMolWt(self.mol)
