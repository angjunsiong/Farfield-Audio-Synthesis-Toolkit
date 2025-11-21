## Materials obtained from public workshop with a simplified mean to simulate phone audio

import numpy as np
import torch
from scipy import signal


# --- Utility: normalize audio for listening without clipping ---
def peak_normalize(x, peak=0.99):
    if isinstance(x, torch.Tensor):
        m = torch.max(torch.abs(x)) + 1e-12
        return (x / m * peak)
    else:
        m = np.max(np.abs(x)) + 1e-12
        return (x / m * peak).astype(np.float32)


def apply_zero_phase_filter(b, a, x):
    # filtfilt minimizes phase distortion — great for offline processing / preprocessing
    return signal.filtfilt(b, a, x).astype(np.float32)


def design_butter_bandpass(low_hz, high_hz, sr, order=6):
    nyq = 0.5 * sr
    Wn = [low_hz / nyq, high_hz / nyq]
    b, a = signal.butter(order, Wn, btype="bandpass", analog=False)
    return b, a


def phone_augment(audio, sr):
    """
    Apply a telephone-like band-pass filter and resample to 16 kHz.
    """

    # Design band-pass to emulate telephone bandwidth
    low_hz, high_hz = 300.0, 3400.0
    b_bp, a_bp = design_butter_bandpass(low_hz, high_hz, sr, order=6)

    # Apply zero-phase band-pass
    x_phone = apply_zero_phase_filter(b_bp, a_bp, audio)
    x_phone = peak_normalize(x_phone)

    # ## Repack x_phone into a torch tensor
    x_phone = torch.from_numpy(x_phone)

    return x_phone, sr
