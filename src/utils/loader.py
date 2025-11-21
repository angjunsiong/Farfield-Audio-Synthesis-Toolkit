import os
import torch
import torchaudio

def load_audio_with_pytorch(
        wav_path: str | os.PathLike,
        target_freq: int | None = 16000,
) -> tuple[torch.Tensor, int]:
    """Loads a WAV file and optionally resamples it to a target frequency.

    This function loads an audio file into a PyTorch tensor with the standard
    [channels, time] format.

    :param wav_path: The path to the WAV audio file.
    :param target_freq: The desired sampling rate. If the audio's native rate
                        is different, it will be resampled. If set to None,
                        no resampling is performed.
    :returns: A tuple containing the audio waveform and its sampling rate.
    """

    # Check if the file exists
    if not os.path.exists(wav_path):
        raise FileNotFoundError(f"Audio file not found: {wav_path}")

    # Load the audio. torchaudio.load defaults to the [channels, time] format.
    waveform, sr = torchaudio.load(wav_path)
    print(f"Audio '{wav_path}' loaded! Native Sampling Rate: {sr}Hz; Shape: {waveform.shape}")

    # Resample only if a target frequency is specified and it differs from the source.
    if target_freq not in (None, sr):
        print(f"Resampling audio from {sr}Hz to {target_freq}Hz...")
        try:
            # Use torchaudio's resample transform for high-quality resampling.
            resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=target_freq,
                                                       resampling_method="sinc_interp_kaiser")
            waveform = resampler(waveform)
            # Update the sample rate to the new target frequency
            sr = target_freq
            print(f"Audio resampled. New shape: {waveform.shape}")
        except Exception as e:
            print(f"Error during resampling: {e}")
            raise

    return waveform, sr