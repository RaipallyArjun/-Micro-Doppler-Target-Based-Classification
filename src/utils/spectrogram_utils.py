import numpy as np
import cv2
from scipy.signal import spectrogram


def signal_to_spectrogram(signal, fs=1000):
    """
    Convert a 1-D signal into a 128x128 normalized spectrogram.
    """

    # Convert to numpy array
    signal = np.asarray(signal, dtype=np.float32)

    # Remove DC component
    signal = signal - np.mean(signal)

    # Generate spectrogram
    frequencies, times, Sxx = spectrogram(
        signal,
        fs=fs,
        window="hann",
        nperseg=256,
        noverlap=200,
        nfft=512,
        mode="magnitude"
    )

    # Convert to decibel scale
    Sxx = 20 * np.log10(Sxx + 1e-8)

    # Normalize
    Sxx = Sxx - np.min(Sxx)

    if np.max(Sxx) != 0:
        Sxx = Sxx / np.max(Sxx)

    # Resize to CNN input size
    Sxx = cv2.resize(
        Sxx,
        (128, 128),
        interpolation=cv2.INTER_LINEAR
    )

    return Sxx.astype(np.float32)


if __name__ == "__main__":

    # Test using a simple sine wave

    t = np.linspace(0, 1, 1000, endpoint=False)

    signal = np.sin(2 * np.pi * 50 * t)

    image = signal_to_spectrogram(signal)

    print("Spectrogram Shape :", image.shape)