## AJS's Fantastic Noise Builder
# This function combines audio-s from a given folder of audio-s
# In this implementation, we randomly pick audios from a designated folder and stack them on top of each other
# Can include options to add sound effects,

# Use Case: This was developed for the Far-Field Audio Synthesis Toolkit (FAST)
# This function serves to combine audio from a folder of noises, toward creating noises that will be synthesised
# with raw speech data, towards the training of noise-robust STT and denoising models

# Formulae for snr_dbs measurement: snr_dbs = 10 log 10 (SNR) = 10 log 10 (P_signal / P_noise)
# >0db means SNR > 1, which means P_signal > P_noise
# Where P is average power

import copy
import os
import random

import torch
import torchaudio.functional as F

from src.audio_effects_new import audio_effector
from src.utils.loader import load_audio_with_pytorch
from src.noise_sizer import noise_sizer


def noise_builder(reference_audio_data,
                  audio_repo,
                  sr=16000,
                  no_of_audio=1,
                  NNR_db_range=(-5, 5),
                  echo=False,
                  tempo_change=False,
                  pitch_shift=False,
                  low_pass=False,
                  no_of_echos_range=(0, 3),
                  echo_delays_range=(50, 150),
                  echo_decays_range=(0.1, 0.2),
                  tempo_range=(0.8, 1.2),
                  pitch_shift_range=(-4, 4),
                  low_pass_order=(2, 5),
                  low_pass_cutoff=(4000, 8000),
                  mode="stationary"
                  ):
    """
    Randomly selects a certain quantity of audio files from a designated folder 
    and superimpose them on top on one another
    Randomly assigns relative signal to noise (SNR... or in this case NNR) ratio from a designated range,
    with options to add effects and perform randomised padding where desired.
    
    Arguments:
    - torch tensor reference_audio_data : The audio data which we wish to add noise on
    - str audio_repo    : The path to the folder containing the noise audio files to be stacked on top of reference_audio_data
    - int no_of_audio   : The number of audios to pick from the folder to be superimposed on each other
    - list NNR_db_range : The range of Noise-to-Noise-Ratio values to be used
                        : The torchaudio function ("add_noise") used for audio stacking considers this value
                        : to be relative signal strength (1st argument) against noise (2nd argument)
    - int sr            : The "enforced" sampling rate for the audio clips to be stacked together
                        : Audio not in this sampling rate will be resampled
    
    » parameters related to src.noise_sizer
    - str mode          : This indicates if we are trying to insent a "stationary" or "non-stationary" noise
                        : (A) If "stationary" (developed for insertion of stationary noise)
                        : - We will truncate the noise (audio_data_2) if it is longer than the reference
                        : - We will loop the noise (audio_data_2) if it is shorter than the reference
                        : (B) If "non-stationary" (developed for insertion of non stationary noise)
                        : - We will insert a (random quantity) of leading zeros to noise (audio_data_2)
                        : Then we will either (1) pad the trailing end of audio_data_2 if the padded audio is shorter than reference audio
                        : - (ii) or truncate audio_data_2 if padded audio_data_2 becomes longer than reference audio
    Returns
    - torch tensor of dimension (1,n_samples), sampling_rate (int)
    
    » parameters to be fed into to src.audio_effects_new
    - bool echo                 : This boolean indicates whether to add echo to the original audio 
    - list no_of_echos_range    : This is a list of 2 int, indicating the lower and upper bounds of the number of echos to be added
    - list echo_delays_range    : This is a list of 2 int, indicating the lower and upper bounds of the delay time
    - list echo_decays_range    : This is a list of 2 floats, indicating the lower and upper bounds of the decay
    - bool tempo                : This boolean indicates whether to effect a change in tempo on the original audio
    - list tempo_range          : This is a list of 2 floats, indicating the lower and upper bound of tempo change
    - bool pitch_shift          : This boolean indicates whether to effect a change in pitch on the original audio
    - int pitch_shift_range     : This is a list of 2 ints, indicating the lower and upper bound of realistic pitch shift
    - bool low_pass             : This boolean indicates whether to effect a low pass filter on the original audio
    - int low_pass_order        : This is a list of 2 ints, indicating the lower and upper bound of the order of low pass
    - int low_pass_cutoff       : This is a list of 2 integers, indicating the lower and upper bound of the critical frequency of the low-pass filter
                                : "which the output signal's power is reduced by half (or its amplitude/pressure for audio by 70.7%)"
    """
    # Initialise list of noise parameters
    noise_paras_dict = {}

    ## Prepare output data in the shape of reference_audio_data
    noise_stack_data = torch.zeros_like(reference_audio_data) + 1e-14

    ## I. Return zero array of same size with reference_audio_data if no_of_audio = 0
    if no_of_audio == 0:
        return noise_stack_data, sr, noise_paras_dict

    ## II. If no_of_audio >= 1; Return data read from randomly-chosen wav file (after applying effect and/or correct-sizing)
    audio_counter = no_of_audio

    while audio_counter >= 1:
        # Initialise dict of noise parameters
        noise_paras = {"noise_name":         None,
                       "effects":            None,
                       "noise_to_stack_NNR": None}

        # Build noise
        noise = random.choice(os.listdir(audio_repo))
        # guard against picking up some weird file
        while "wav" not in noise:
            # TODO: hangs permanently if noise folder is empty
            # TODO: brittle; e.g., accepts `not-a-wav-file.txt`
            noise = random.choice(os.listdir(audio_repo))
        noise_path = os.path.join(audio_repo, noise)
        noise_data, sr_noise = load_audio_with_pytorch(noise_path)

        # Add sound effects
        noise_data, sr_noise, paras = audio_effector(noise_data, sr_noise,
                                                     echo=echo,
                                                     tempo_change=tempo_change,
                                                     pitch_shift=pitch_shift,
                                                     low_pass=low_pass,
                                                     no_of_echos_range=no_of_echos_range,
                                                     echo_delays_range=echo_delays_range,
                                                     echo_decays_range=echo_decays_range,
                                                     tempo_range=tempo_range,
                                                     pitch_shift_range=pitch_shift_range,
                                                     low_pass_order=low_pass_order,
                                                     low_pass_cutoff=low_pass_cutoff
                                                     )

        # Size data to match that of reference audio
        noise_data, pad_size = noise_sizer(reference_audio_data, noise_data, mode=mode)

        # Log parameters
        noise_paras["noise_name"] = noise
        noise_paras["effects"] = paras
        noise_paras["pad_size"] = pad_size

        # SCENARIO 1: no_of_audio == 1
        ## if there is only 1 loop to begin with, we return noise_data and sr_noise directly
        if no_of_audio == 1:
            # Log parameter
            noise_paras_dict[f"noise_{no_of_audio - audio_counter + 1}"] = noise_paras

            return noise_data, sr, noise_paras_dict

        # SCENARIO 2: no_of_audio > 1
        # if no_of_audio is > 1, then we perform loop the stacking of audio
        ## for the first loop, we initialise noise_data as the "base" of the stack

        if no_of_audio == audio_counter:  # checker for loop number 1, setting up "base" of the stack
            noise_stack_data = copy.deepcopy(noise_data)

        # for loops beyond the first, we stack the noise data onto the "base"
        else:
            # generate a random SNR
            noise_to_stack_ratio_dbs = random.uniform(NNR_db_range[0], NNR_db_range[1])
            # stack audio data
            noise_stack_data = F.add_noise(noise_stack_data, noise_data, torch.tensor([noise_to_stack_ratio_dbs]))

            # Log parameter
            noise_paras["noise_to_stack_NNR"] = noise_to_stack_ratio_dbs

        # Log noise_paras
        noise_paras_dict[f"noise_{no_of_audio - audio_counter + 1}"] = noise_paras

        audio_counter -= 1

    ## Return noise stack when we are done!
    return noise_stack_data, sr, noise_paras_dict
