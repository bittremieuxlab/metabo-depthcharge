class Molecule:
    """Base molecule object.

    Parameters
    ----------
    smiles : str
        SMILES string identifying the molecule.
    metadata : dict, optional
        Arbitrary per-molecule key/value pairs (inchi, name, source,
        biological activity labels, ...). Defaults to an empty dict.
    """

    def __init__(self, smiles: str, metadata: dict | None = None):
        self.smiles = smiles
        self.metadata = metadata if metadata is not None else {}

    def __repr__(self) -> str:
        return f"Molecule(smiles={self.smiles!r}, metadata={self.metadata!r})"
