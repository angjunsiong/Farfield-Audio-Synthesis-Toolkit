"""
written by gemini, TODO: check correctness
we should probably standardize on torchaudio end to end, but rubberband only works on arrays
hence this simple wrapper to convert back and forth
might be worth generalizing it if we have a lot of things that only work on arrays
but ideally we stick to torch transforms and keep things on gpu
"""

import logging
import numpy as np
import pyrubberband
import torch


def rubberband_stretch_tensor(waveform: torch.Tensor, rate: float, sr: int = 16000) -> torch.Tensor:
    """
    Wraps pyrubberband to handle Torch Tensors directly.
    Input: [Channels, Time] or [Time]
    Output: [Channels, New_Time]
    """
    # 1. Check if GPU, move to CPU if necessary (rubberband is CPU only)
    was_cuda = waveform.is_cuda
    if was_cuda:
        waveform = waveform.cpu()

    # 2. Convert to Numpy
    # PyRubberband expects [Time] or [Time, Channels]
    # Torchaudio is [Channels, Time]
    wav_np = waveform.numpy()

    # Handle dimensions for pyrubberband
    if wav_np.ndim == 2:
        wav_np = wav_np.T  # Transpose to [Time, Channels]

    # 3. Apply Rubberband
    try:
        # pyrubberband returns numpy array
        stretched_np = pyrubberband.time_stretch(wav_np, sr, rate)
    except Exception:
        logging.exception('Rubberband failed, returning original')
        return waveform.cuda() if was_cuda else waveform

    # 4. Convert back to Tensor and Restore Dimensions
    if stretched_np.ndim == 2:
        stretched_np = stretched_np.T  # Transpose back to [Channels, Time]

    stretched_tensor = torch.from_numpy(stretched_np).float()

    # 5. Move back to GPU if it started there
    if was_cuda:
        stretched_tensor = stretched_tensor.cuda()

    return stretched_tensor
