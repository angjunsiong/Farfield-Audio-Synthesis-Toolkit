import os

import librosa
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchaudio


def plot_waveform(waveform, sample_rate, title="Waveform", xlim=None):
    waveform = waveform.numpy()

    num_channels, num_frames = waveform.shape
    time_axis = torch.arange(0, num_frames) / sample_rate

    figure, axes = plt.subplots(num_channels, 1)
    if num_channels == 1:
        axes = [axes]

    for c in range(num_channels):
        axes[c].plot(time_axis, waveform[c], linewidth=1)
        axes[c].grid(True)
        if num_channels > 1:
            axes[c].set_ylabel(f"Channel {c + 1}")
        if xlim:
            axes[c].set_xlim(xlim)
    figure.suptitle(title)


def plot_spectrogram(audio_path,
                     xlim=None,
                     threshold=False,
                     threshold_factor=3,
                     # hop_length defaulted to 32 › default value will make spec very low in time-resolution
                     hop_length=32):
    # load audio, convert to freq domain
    waveform, sampling_rate = librosa.load(audio_path, sr=None)
    spec_wave = np.abs(librosa.stft(waveform, hop_length=hop_length))

    # prepare plot
    fig, ax = plt.subplots()

    # checks if threshold is to be applied ; apply threshold here
    if threshold == True:
        spec_wave[spec_wave < np.mean(spec_wave) * threshold_factor] = 0

    # publish plot
    S_db = librosa.amplitude_to_db(np.abs(spec_wave), ref=np.max)
    img = librosa.display.specshow(S_db,
                                   sr=sampling_rate,
                                   x_axis="s",
                                   y_axis="linear",
                                   ax=ax,
                                   hop_length=hop_length)
    ax.set(title=f"Spectrogram for {audio_path.split('/')[-1]}")
    ax.set_xlim(xlim)
    fig.colorbar(img, ax=ax, format="%+2.f dB")


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
