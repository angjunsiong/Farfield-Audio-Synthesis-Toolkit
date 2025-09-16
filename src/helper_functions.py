import matplotlib.pyplot as plt
import random
import torch
import torchaudio
import torchaudio. functional as F
import librosa
import numpy as np
import os

def plot_waveform(waveform, sample_rate, title="Waveform", xlim=None):
    waveform = waveform.numpy()

    num_channels, num_frames = waveform. shape
    time_axis = torch.arange(0, num_frames) / sample_rate

    figure, axes = plt. subplots(num_channels, 1)
    if num_channels == 1:
        axes = [axes]

    for c in range (num_channels) :
        axes[c].plot(time_axis, waveform[c], linewidth=1)
        axes[c].grid(True)
        if num_channels > 1:
            axes[c].set_ylabel(f"Channel {c+1}")
        if xlim:
            axes[c].set_xlim(xlim)
    figure.suptitle(title)

def plot_specgram(audio_path,
                  xlim=None,
                  threshold = False,
                  threshold_factor = 3,
                  #hop_length defaulted to 32 › default value will make spec very low in time-resolution
                  hop_length=32):
    
    #load audio, convert to freq domain
    waveform, sampling_rate = librosa. load(audio_path, sr=None)
    spec_wave = np.abs(librosa.stft(waveform, hop_length=hop_length))

    # prepare plot
    fig, ax = plt. subplots()

    # checks if threshold is to be applied ; apply threshold here
    if threshold == True:
        spec_wave[spec_wave<np.mean(spec_wave)*threshold_factor] = 0

    # publish plot
    S_db = librosa.amplitude_to_db(np.abs(spec_wave), ref=np.max)
    img = librosa.display.specshow(S_db,
                                   sr = sampling_rate,
                                   x_axis = "s",
                                   y_axis = "linear",
                                   ax=ax,
                                   hop_length=hop_length)
    ax.set(title=f"Spectrogram for {audio_path.split('/')[-1]}")
    ax.set_xlim(xlim)
    fig.colorbar(img, ax=ax, format="%+2.f dB")

def load_audio_with_pytorch(wav_path, convert_freq = True, target_freq = 16000):
    # Defaulted to 16k resampling
    # channels_firs (bool): if true return Tensor with dimension [channel, time], else [time, channel]
    waveform, sr = torchaudio.load(wav_path, channels_first=False)
    print(f"Audio {wav_path} loaded!; Native Sampling_Rate: {sr}Hz; Shape: {waveform.shape}")
    if sr != target_freq and convert_freq == True:
        # Transposed because resample function resamples along last axis
        # sinc interpolation kaiser window method offers higher quality, lower speed resampling
        waveform = torchaudio.functional.resample(waveform.T, orig_freq=sr, new_freq=target_freq,resampling_method="sinc_interp_kaiser")
        sr = target_freq
        print(f"Audio {wav_path} resampled to {target_freq}Hz")
        return waveform, sr

    else:
        return waveform.T, sr