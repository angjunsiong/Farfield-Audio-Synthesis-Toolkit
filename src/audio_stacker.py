# AJS's not-worth-mentioniong Audio Stacker
# This function combines audio/noise and noise given a SNR

import torchaudio.functional as F
import torch
import random

def audio_noise_stack(audio_1_data,
                      audio_2_data,
                      SNR
                      ):
    
    if (audio_1_data.shape != audio_2_data.shape):
        raise ValueError ("The shape of the audio and noise data do not match!")
    
    stack_data = F.add_noise(audio_1_data, audio_2_data, torch.tensor([SNR]))
    
    # normalise audio after this
    max_value = stack_data.abs().max()
    if max_value > 0:
        stack_data = stack_data/max_value

    return stack_data