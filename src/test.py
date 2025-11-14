import torch
import numpy as np
import matplotlib.pyplot as plt
from helper_functions import plot_waveform, plot_spectrogram

SAMPLE_RATE = 44100
duration = 1
t = torch.linspace(0., duration, int(SAMPLE_RATE * duration), dtype=torch.float32)
mono_wave = 0.5 * torch.sin(2 * torch.pi * 440 * t) # 440 Hz tone

print(f"Plotting mono wave with shape: {mono_wave.shape}")
plot_waveform(mono_wave, SAMPLE_RATE, title="Mono Waveform (440 Hz)")
plot_spectrogram(mono_wave, SAMPLE_RATE, title="Mono Waveform (440 Hz)")


# 2. Generate a sample 2-second stereo waveform
duration = 2
t = torch.linspace(0., duration, int(SAMPLE_RATE * duration), dtype=torch.float32)
ch1 = 0.5 * torch.sin(2 * torch.pi * 220 * t) # 220 Hz tone
ch2 = 0.5 * torch.sin(2 * torch.pi * 330 * t) # 330 Hz tone
stereo_wave = torch.stack([ch1, ch2])

print(f"Plotting stereo wave with shape: {stereo_wave.shape}")
plot_waveform(stereo_wave, SAMPLE_RATE, title="Stereo Waveform")
plot_spectrogram(stereo_wave, SAMPLE_RATE, title="Stereo Waveform")

# 3. Plot with xlim
print("Plotting stereo wave with x-limit")
plot_waveform(stereo_wave, SAMPLE_RATE, title="Stereo Waveform (Zoomed)", xlim=(0, 0.05))
plot_spectrogram(stereo_wave, SAMPLE_RATE, title="Stereo Waveform", xlim=(0, 0.05))