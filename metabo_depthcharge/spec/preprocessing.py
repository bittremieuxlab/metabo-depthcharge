import matplotlib.pyplot as plt
import numpy as np


class Spectrum:
    """Base spectrum object.

    Parameters
    ----------
    mz : 1-D np.array, optional
        mz values, by default None
    intensity : 1-D np.array, optional
        intensity values, by default None
    metadata : dict, optional
        Arbitrary per-spectrum key/value pairs (adduct, collision energy,
        instrument type, …), by default empty dict.

    Example
    -------
    .. code-block:: python

        spec = Spectrum(
            mz=np.array([105.0, 150.0]),
            intensity=np.array([0.5, 1.0]),
        )
        len(spec)  # 2
    """

    def __init__(self, mz=None, intensity=None, metadata=None):
        self.mz = mz
        self.intensity = intensity
        self.metadata = metadata if metadata is not None else {}
        if self.intensity is not None:
            if np.issubdtype(self.intensity.dtype, np.unsignedinteger):
                self.intensity = self.intensity.astype(int)
        if self.mz is not None:
            if np.issubdtype(self.mz.dtype, np.unsignedinteger):
                self.mz = self.mz.astype(int)

    def __getitem__(self, index):
        return Spectrum(mz=self.mz[index], intensity=self.intensity[index])

    def __len__(self):
        if self.mz is not None:
            return self.mz.shape[0]
        else:
            return 0

    def plot(self, as_peaks=True, **kwargs):
        """Plot a spectrum via matplotlib.

        Parameters
        ----------
        as_peaks : bool, optional
            Draw peaks as individual vertical lines instead of connecting
            points, by default True.
        """
        if as_peaks:
            mz_plot = np.stack([self.mz - 0.001, self.mz, self.mz + 0.001]).T.reshape(
                -1
            )
            int_plot = np.stack(
                [
                    np.zeros_like(self.intensity),
                    self.intensity,
                    np.zeros_like(self.intensity),
                ]
            ).T.reshape(-1)
        else:
            mz_plot, int_plot = self.mz, self.intensity
        plt.plot(mz_plot, int_plot, **kwargs)

    def __repr__(self):
        string_ = np.array2string(
            np.stack([self.mz, self.intensity]), precision=5, threshold=10, edgeitems=2
        )
        mz_string, int_string = string_.split("\n")
        mz_string = mz_string[1:]
        int_string = int_string[1:-1]
        return "Spectrum([\n\tmz  = %s,\n\tint = %s\n])" % (mz_string, int_string)

    def torch(self):
        """Return the spectrum intensity and mz values as a dict of torch tensors."""
        import torch

        return {
            "mz": torch.tensor(self.mz).unsqueeze(0).to(torch.float64),
            "intensity": torch.tensor(self.intensity).unsqueeze(0).to(torch.float32),
        }


class Normalizer:
    """Normalize spectrum intensity so that the maximum intensity equals 1.

    Example
    -------
    .. code-block:: python

        spectrum_processed = Normalizer()(spectrum)
    """

    def __init__(self):
        pass

    def __call__(self, spectrum):
        return Spectrum(
            intensity=spectrum.intensity / spectrum.intensity.max(),
            mz=spectrum.mz,
            metadata=getattr(spectrum, "metadata", {}),
        )


class Trimmer:
    """Remove peaks outside a given m/z range.

    Parameters
    ----------
    min : int, optional
        Remove peaks with mz below this value, by default 0.
    max : int, optional
        Remove peaks with mz above this value, by default 2000.

    Example
    -------
    .. code-block:: python

        spectrum_processed = Trimmer(min=50, max=1000)(spectrum)
    """

    def __init__(self, min=0, max=2000):
        self.range = [min, max]

    def __call__(self, spectrum):
        indices = (self.range[0] < spectrum.mz) & (spectrum.mz < self.range[1])
        return Spectrum(
            intensity=spectrum.intensity[indices],
            mz=spectrum.mz[indices],
            metadata=getattr(spectrum, "metadata", {}),
        )


class PeakFilter:
    """Filter peaks by count and/or minimum intensity.

    Parameters
    ----------
    max_number : int, optional
        Keep only the top-N most intense peaks, by default None (no limit).
    min_intensity : float, optional
        Discard peaks below this intensity, by default None (no limit).

    Example
    -------
    .. code-block:: python

        spectrum_processed = PeakFilter(max_number=128, min_intensity=0.01)(spectrum)
    """

    def __init__(self, max_number=None, min_intensity=None):
        self.max_number = max_number
        self.min_intensity = min_intensity

    def __call__(self, spectrum):
        s = Spectrum(
            intensity=spectrum.intensity,
            mz=spectrum.mz,
            metadata=getattr(spectrum, "metadata", {}),
        )

        if self.max_number is not None:
            indices = np.argsort(-s.intensity, kind="stable")
            take = np.sort(indices[: self.max_number])
            s.mz = s.mz[take]
            s.intensity = s.intensity[take]

        if self.min_intensity is not None:
            take = s.intensity >= self.min_intensity
            s.mz = s.mz[take]
            s.intensity = s.intensity[take]

        return s


class CollapseSteppedCE:
    """Collapse suffixed CE key in ``"metadata"`` into a single ``collision_energy`` key.

    Some MGFs have stepped collision energy as ``COLLISION_ENERGY_1``,
    ``COLLISION_ENERGY_2``. Or ``NORMALIZED_COLLISION_ENERGY_N`` variants.

    Parameters
    ----------
    source : {"normalized", "absolute"}, default "normalized"
        Which family to collapse. ``"normalized"`` matches
        ``normalized_collision_energy_N``; ``"absolute"`` matches
        ``collision_energy_N``. Pyteomics lowercases MGF keys.

    See Also
    --------
    ~metabo_depthcharge.spec.metadata_parsers.encode_collision_energy :
        Downstream row-wise parser that turns the collapsed
        ``collision_energy`` field into the ``float32`` value consumed by
        ``MetadataEncoder``.

    Example
    -------
    .. code-block:: python

        spectrum.metadata = {"normalized_collision_energy_1": 20, "normalized_collision_energy_2": 40}
        CollapseSteppedCE()(spectrum).metadata["collision_energy"]  # 30.0
    """

    def __init__(self, source: str = "normalized"):
        if source not in ("normalized", "absolute"):
            raise ValueError(
                f"source must be 'normalized' or 'absolute', got {source!r}"
            )
        self._base = (
            "normalized_collision_energy"
            if source == "normalized"
            else "collision_energy"
        )

    def __call__(self, spectrum):
        vals = []
        for k, v in spectrum.metadata.items():
            kl = k.lower()
            if kl != self._base and not kl.startswith(self._base + "_"):
                continue
            try:
                f = float(v)
            except (TypeError, ValueError):
                continue
            if not np.isnan(f):
                vals.append(f)
        if vals:
            spectrum.metadata["collision_energy"] = sum(vals) / len(vals)
        return spectrum


class SequentialPreprocessor:
    """Chain multiple preprocessors into a single callable pipeline.

    See Also
    --------
    ~metabo_depthcharge.spec.preprocessing.DefaultSpectrumProcessor :
        Ready-made instance of this class (Trimmer + PeakFilter +
        Normalizer). Plug it into the ``processor`` argument of
        :class:`~metabo_depthcharge.datasets.SpectrumDataset` to bake
        this preprocessing into a dataset.

    Example
    -------
    .. code-block:: python

        preprocessor = SequentialPreprocessor(
            Trimmer(min=0, max=2000),
            PeakFilter(max_number=128),
            Normalizer(),
        )
        spectrum_processed = preprocessor(spectrum)
    """

    def __init__(self, *args):
        self.preprocessors = args

    def __call__(self, spectrum):
        for step in self.preprocessors:
            spectrum = step(spectrum)
        return spectrum


#: Default preprocessing pipeline. Trim to ``[0, 2000]`` m/z, cap to the 128
#: most intense peaks, then normalize. A ready-to-use
#: :class:`SequentialPreprocessor` instance; call it on a :class:`Spectrum`.
DefaultSpectrumProcessor = SequentialPreprocessor(
    Trimmer(min=0, max=2000),
    PeakFilter(max_number=128),
    Normalizer(),
)
