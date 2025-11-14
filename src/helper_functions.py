##### Should probably add fig.show() or plt.show() so that multiple can be plotted in the same cell

import os

import librosa
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchaudio
from typing import Optional, Literal


def plot_waveform(waveform: torch.Tensor | np.ndarray, sample_rate: int, title: str = "Waveform", xlim: Optional[tuple] = None):
    """
    Plots a waveform tensor.

    Handles both mono (1D) and multi-channel (2D) tensors.

    Args:
        waveform (torch.Tensor | np.ndarray): The audio waveform. 
            Expected shapes:
            - (num_frames,): Mono audio
            - (num_channels, num_frames): Multi-channel audio
        sample_rate (int): The sample rate of the audio.
        title (str, optional): The title for the plot. Defaults to "Waveform".
        xlim (tuple, optional): A (min, max) tuple to set the x-axis limits. Defaults to None.
    """
    
    # --- 1. Waveform Checking ---
    if isinstance(waveform, torch.Tensor):
        waveform = waveform.numpy()
    
    if not isinstance(waveform, np.ndarray):
        raise TypeError(f"Input must be a torch.Tensor or np.ndarray, but got {type(waveform)}")

    if waveform.ndim == 1:
        waveform = waveform.reshape(1, -1) # Mono case: unsqueeze to (1, num_frames)
    elif waveform.ndim > 2:
        # Unsupported case (e.g., batch)
        raise ValueError(
            f"Waveform must be 1D (num_frames) or 2D (num_channels, num_frames), "
            f"but got shape {waveform.shape}"
        )
    
    # --- 2. Data Preparation ---
    num_channels, num_frames = waveform.shape
    time_axis = np.arange(0, num_frames) / sample_rate

    # --- 3. Plotting ---
    figure, axes = plt.subplots(num_channels, 1, figsize=(10, 2 * num_channels), sharex=True)
    
    if num_channels == 1:
        # Ensure 'axes' is always iterable (a list)
        axes = [axes]

    for c in range(num_channels):
        axes[c].plot(time_axis, waveform[c], linewidth=1)
        axes[c].grid(True)
        if num_channels > 1:
            axes[c].set_ylabel(f"Channel {c + 1}")
        if xlim:
            axes[c].set_xlim(xlim)
            
    # Add a shared x-axis label
    axes[-1].set_xlabel("Time (s)")
    
    figure.suptitle(title)
    
    # Add tight_layout for better spacing
    figure.tight_layout()
    
    # --- 4. Display Plot ---
    plt.show()


def plot_spectrogram(
    waveform: torch.Tensor | np.ndarray,
    sample_rate: int,
    title: str = "Spectrogram",
    n_fft: int = 2048,
    hop_length: int = 512,
    win_length: Optional[int] = None,
    window: str = 'hann',
    to_db: bool = True,
    y_axis: Literal['linear', 'log'] = 'log',
    xlim: Optional[tuple] = None
):
    """
    Plots a spectrogram for a given waveform.

    Handles both mono (1D) and multi-channel (2D) inputs, which can be
    either torch.Tensors or np.ndarrays. Multi-channel inputs are plotted
    as separate subplots.

    Args:
        waveform (torch.Tensor | np.ndarray): The audio waveform.
            Expected shapes:
            - (num_frames,): Mono audio
            - (num_channels, num_frames): Multi-channel audio
        sample_rate (int): The sample rate of the audio (e.g., 44100).
        title (str, optional): The title for the overall figure. Defaults to "Spectrogram".
        n_fft (int, optional): The number of FFT components. Defaults to 2048.
        hop_length (int, optional): The number of samples between STFT frames. Defaults to 512.
        win_length (int, optional): Window length. Defaults to n_fft if None.
        window (str, optional): The window function (e.g., 'hann'). Defaults to 'hann'.
        to_db (bool, optional): Whether to convert amplitude to decibels. Defaults to True.
        y_axis (Literal['linear', 'log'], optional): The frequency axis scale. Defaults to 'log'.
        xlim (Optional[tuple], optional): A (min, max) tuple in seconds to set the x-axis (time) limits. Defaults to None.
    """
    
    # --- 1. Input Handling & Dimension Checking ---
    
    # Convert to numpy if it's a torch tensor
    if isinstance(waveform, torch.Tensor):
        waveform = waveform.numpy()

    if not isinstance(waveform, np.ndarray):
        raise TypeError(f"Input must be a torch.Tensor or np.ndarray, but got {type(waveform)}")

    # Standardize dimensions to 2D (num_channels, num_frames)
    if waveform.ndim == 1:
        # Mono case: reshape to (1, num_frames)
        waveform = waveform.reshape(1, -1)
    elif waveform.ndim > 2:
        # Unsupported case (e.g., batch)
        raise ValueError(
            f"Waveform must be 1D (num_frames) or 2D (num_channels, num_frames), "
            f"but got shape {waveform.shape}"
        )
    
    num_channels, num_frames = waveform.shape

    # --- 2. Plotting Setup ---
    # Create subplots for each channel
    # Note: sharex is True, so xlim will apply to all subplots
    figure, axes = plt.subplots(num_channels, 1, figsize=(10, 4 * num_channels), sharex=True)
    
    if num_channels == 1:
        # Ensure 'axes' is always iterable (a list)
        axes = [axes]

    # --- 3. STFT and Plotting Loop ---
    for c in range(num_channels):
        ax = axes[c]
        
        # Get the waveform for the current channel
        channel_wave = waveform[c]

        # 3a. Calculate STFT
        # librosa.stft returns a complex-valued matrix
        S_complex = librosa.stft(
            channel_wave,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            window=window
        )
        
        # 3b. Get magnitude
        S_mag = np.abs(S_complex)
        
        # 3c. Convert to decibels (if requested)
        plot_data = S_mag
        colorbar_format = None
        
        if to_db:
            # librosa.amplitude_to_db is a safe way to convert
            plot_data = librosa.amplitude_to_db(S_mag, ref=np.max)
            colorbar_format = '%+2.0f dB'

        # 3d. Plot the spectrogram using librosa.display.specshow
        # This automatically handles axis labels (Time, Hz)
        img = librosa.display.specshow(
            plot_data,
            ax=ax,
            sr=sample_rate,
            hop_length=hop_length,
            x_axis='time',
            y_axis=y_axis
        )
        
        # 3e. Add labels and colorbar
        if num_channels > 1:
            ax.set_ylabel(f"Channel {c + 1}")
            
        figure.colorbar(img, ax=ax, format=colorbar_format)

    # --- 4. Final Touches ---
    
    # Set xlim if provided. Since sharex=True, this applies to all subplots.
    if xlim:
        axes[0].set_xlim(xlim) # Set on one axis, it will apply to all

    figure.suptitle(title)
    
    # Use tight_layout for better spacing
    figure.tight_layout()
    
    # Display the plot
    plt.show()
