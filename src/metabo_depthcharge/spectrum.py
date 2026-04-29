import numpy as np
import matplotlib.pyplot as plt
import torch


class SpectrumObject:
    """Base Spectrum Object class

    Can be instantiated directly with 1-D np.arrays for mz and intensity.

    Parameters
    ----------
    mz : 1-D np.array, optional
        mz values, by default None
    intensity : 1-D np.array, optional
        intensity values, by default None
    """

    def __init__(self, mz=None, intensity=None):
        self.mz = mz
        self.intensity = intensity
        if self.intensity is not None:
            if np.issubdtype(self.intensity.dtype, np.unsignedinteger):
                self.intensity = self.intensity.astype(int)
        if self.mz is not None:
            if np.issubdtype(self.mz.dtype, np.unsignedinteger):
                self.mz = self.mz.astype(int)

    def __getitem__(self, index):
        return SpectrumObject(mz=self.mz[index], intensity=self.intensity[index])

    def __len__(self):
        if self.mz is not None:
            return self.mz.shape[0]
        else:
            return 0

    def plot(self, as_peaks=False, **kwargs):
        """Plot a spectrum via matplotlib

        Parameters
        ----------
        as_peaks : bool, optional
            draw points in the spectrum as individualpeaks, instead of connecting the points in the spectrum, by default False
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
        return "SpectrumObject([\n\tmz  = %s,\n\tint = %s\n])" % (mz_string, int_string)
    
    def torch(self):
        """Converts spectrum to dict of tensors"""
        return {
            "mz": torch.tensor(self.mz).unsqueeze(0).to(torch.float64),
            "intensity": torch.tensor(self.intensity).unsqueeze(0).to(torch.float32),
        }
    
class Normalizer:
    """Pre-processing function for normalizing the intensity of a spectrum.
    Max intensity will be 1.
    """

    def __init__(self):
        pass

    def __call__(self, SpectrumObj):
        s = SpectrumObject()

        s = SpectrumObject(
            intensity=SpectrumObj.intensity / SpectrumObj.intensity.max(),
            mz=SpectrumObj.mz,
        )
        return s
    

class Trimmer:
    """Pre-processing function for trimming ends of a spectrum.
    This can be used to remove inaccurate measurements.

    Parameters
    ----------
    min : int, optional
        remove all measurements with mz's lower than this value, by default 0
    max : int, optional
        remove all measurements with mz's higher than this value, by default 2000
    """

    def __init__(self, min=0, max=2000):
        self.range = [min, max]

    def __call__(self, SpectrumObj):
        indices = (self.range[0] < SpectrumObj.mz) & (SpectrumObj.mz < self.range[1])

        s = SpectrumObject(
            intensity=SpectrumObj.intensity[indices], mz=SpectrumObj.mz[indices]
        )
        return s

class PeakFilter:
    """Pre-processing function for filtering peaks.

    Filters in two ways: absolute number of peaks and height.

    Parameters
    ----------
    max_number : int, optional
        Maximum number of peaks to keep. Prioritizes peaks to keep by height.
        by default None, for no filtering
    min_intensity : float, optional
        Min intensity of peaks to keep, by default None, for no filtering
    """

    def __init__(self, max_number=None, min_intensity=None):
        self.max_number = max_number
        self.min_intensity = min_intensity

    def __call__(self, SpectrumObj):
        s = SpectrumObject(intensity=SpectrumObj.intensity, mz=SpectrumObj.mz)

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
    """Chain multiple preprocessors so that a pre-processing pipeline can be called with one line.

    Example:
    ```python
    preprocessor = SequentialPreprocessor(
        Normalizer(),
        PeakFilter(max_number = 128)
    )
    preprocessed_spectrum = preprocessor(spectrum)
    ```
    """

    def __init__(self, *args):
        self.preprocessors = args

    def __call__(self, SpectrumObj):
        for step in self.preprocessors:
            SpectrumObj = step(SpectrumObj)
        return SpectrumObj

DefaultSpectrumProcessor = SequentialPreprocessor(
    Trimmer(min=0, max=2000),
    PeakFilter(max_number=128),
    Normalizer()
)
