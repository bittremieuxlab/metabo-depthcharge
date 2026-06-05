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

    def plot(self, as_peaks=False, **kwargs):
        """Plot a spectrum via matplotlib.

        Parameters
        ----------
        as_peaks : bool, optional
            Draw peaks as individual vertical lines instead of connecting
            points, by default False.
        """
        if as_peaks:
            mz_plot = np.stack([self.mz - 1, self.mz, self.mz + 1]).T.reshape(-1)
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
        """Return the spectrum as a dict of torch tensors."""
        import torch

        return {
            "mz": torch.tensor(self.mz).unsqueeze(0).to(torch.float64),
            "intensity": torch.tensor(self.intensity).unsqueeze(0).to(torch.float32),
        }


class Normalizer:
    """Normalize spectrum intensity so that the maximum intensity equals 1."""

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


class SequentialPreprocessor:
    """Chain multiple preprocessors into a single callable pipeline.

    Example
    -------
    >>> preprocessor = SequentialPreprocessor(
    ...     Trimmer(min=0, max=2000),
    ...     PeakFilter(max_number=128),
    ...     Normalizer(),
    ... )
    >>> processed = preprocessor(spectrum)
    """

    def __init__(self, *args):
        self.preprocessors = args

    def __call__(self, spectrum):
        for step in self.preprocessors:
            spectrum = step(spectrum)
        return spectrum


DefaultSpectrumProcessor = SequentialPreprocessor(
    Trimmer(min=0, max=2000),
    PeakFilter(max_number=128),
    Normalizer(),
)
