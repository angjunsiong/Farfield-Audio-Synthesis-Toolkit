## AJS's Can-Lah Audio Effects... Effector!
## This is my 2nd try after I can't get torchaudio to work; implementing effects using pyrubberband and math instead
## Of note, opensource research suggest pyrubberband is superior to librosa for timestretch
## Typical complaints "because librosa implements it using stft/istft and phase signal is lost, leading to clangy sounding echos"

# This function creates audio effects of echo and tempo change for audios
# This was originally developed for the Far-Field Audio Synthesis Toolkit (FAST)
# It serves to enhance the data augmentation process for background noises by introducing echo and tempo change
# Note: The range of the parameters were determined using my own hearing tests

import random
import pyrubberband
import numpy as np
import torch
from scipy import signal

def audio_effector(audio_wav,
                   sr=16000,
                   echo = False, 
                   tempo_change = False,
                   pitch_shift=False,
                   low_pass = False,
                   no_of_echos_range = (0,3),
                   echo_delays_range = (50, 150),
                   echo_decays_range = (0.1, 0.2),
                   list_of_delays = None,
                   list_of_decays = None,
                   tempo_range = (0.8,1.2),
                   pitch_shift_range = (-4,4),
                   low_pass_order = (2,5),
                   low_pass_cutoff = (2000, 8000)
                   ):
    """
    Arguments:
    - tensor    audio_wav           : This is waveform data of the original audio, after reading audio with torchaudio
    - int       sr                  : This is the sampling rate of audio_wav
    - bool      echo                : This boolean indicates whether to add echo to the original audio audio_wav
    - list      no_of_echos_range   : This is a list of 2 int, indicating the lower and upper bounds of the number of echos, in milliseconds
    - list      echo_delays_range   : This is a list of 2 int, indicating the lower and upper bounds of the delay time
    - list      echo_decays_range   : This is a list of 2 floats, indicating the lower and upper bounds of the decay
    - list      list_of_delays      : A prescribed list of delays; Used for audio regeneration
    - list      list_of_decays      : A prescribed list of decays; Used for audio regeneration
    - bool      tempo               : This boolean indicates whether to effect a change in tempo on the original audio
    - list      tempo_range         : This is a list of 2 floats, indicating the lower and upper bound of tempo change
    - bool      pitch_shift         : This boolean indicates whether to effect a change in pitch on the original audio
    - int       pitch_shift_range   : This is a list of 2 ints, indicating the lower and upper bound of realistic pitch shift
    - bool      low_pass            : This boolean indicates whether to effect a low pass fi l t e r on the original audio
    - int       low_pass_order      : This is a list of 2 ints, indicating the lower and upper bound of the order of the low pass cutoff
    - int       low_pass_cutoff     : This is a list of 2 ints, indicating the lower and upper bound of the critical frequency
    
    Return: 
    - torch tensor  (1,n_samples), 
    - int           sampling_rate (int)
    - dict          parameters  
    """


    # convert to 1-D (transposed) numpy array if wav audio is not yet in numpy; ensures compatibility with all functions
    ## We have to do this because pyrubberband is designed to worked with soundfile, which opens audio as a 1-D numpy array!
    ## On the other hand torchaudio opens as a 2-D tensor (data, channel#) or (channel#, data) 
    ## Here we convert the 2-D torch tensor into the 1-D array

    if not isinstance(audio_wav, np.ndarray):
        audio_wav = audio_wav.numpy()[0]

    # Set up parameters log
    paras = {}

    # I. Implement Echo
    # determine number of echos
    if list_of_delays is None:
        echo_counter = random.randint(no_of_echos_range[0], no_of_echos_range[1])
    else:
        echo_counter = len(list_of_delays)
    
    if echo and echo_counter!=0:
        # sort delays (in descending order) and decays (in descending order) to get more realistic echos
        #??? !!! was using randint earlier to get the milliseconds in delay 
        if list_of_delays is None:
            list_of_delays = [(random.uniform(echo_delays_range[0], echo_delays_range[1])*(i+1)/1000) for i in range (echo_counter)]
            list_of_delays.sort()
            list_of_decays = [random.uniform(echo_decays_range[0], echo_decays_range[1]) for i in range(echo_counter)]
            list_of_decays.sort()

        # loop through N times, to stack N different audio
        for count in range(echo_counter):
            echo_data = echo_generator(audio_wav, sr, list_of_delays[count], list_of_decays[count])
            audio_wav += echo_data

        # log parameters
        paras["list_of_delays"] = list_of_delays
        paras["list_of_decays"] = list_of_decays

    # II. Implement change of temp
    ## Effect tempo change
    
    if tempo_change:
        tempo_change_rate = random.uniform(tempo_range[0], tempo_range[1])
        audio_wav = pyrubberband.pyrb.time_stretch(audio_wav, 
                                                   sr=sr,
                                                   rate=tempo_change_rate)
        
        #log parameters
        paras["tempo_change_rate"] = tempo_change_rate

    # III. Implement pitchshift
    if pitch_shift:
        n_steps=random.randint(pitch_shift_range[0],pitch_shift_range[1])
        audio_wav = pyrubberband.pyrb.pitch_shift(audio_wav, sr=sr, n_steps=n_steps)
        
        paras["pitch_shift"] = n_steps
    
    # IV. Implement low-pass filter (using a butterworth filter)
    if low_pass:
        # Design butterworth filter
        low_pass_order = random.randint(low_pass_order[0],low_pass_order[1])
        low_pass_cutoff = random.randint(low_pass_cutoff[0], low_pass_cutoff[1])        
        
        b, a = signal.butter(N=low_pass_order,
                             Wn=low_pass_cutoff,
                             btype="low",
                             analog=False,
                             fs= sr)
        # Apply filter
        audio_wav = signal.lfilter(b, a, audio_wav)

        # log parameters; convert to list for json-ification
        paras["low_pass"] = {"low_pass_order": low_pass_order,
                             "low_pass_cutoff": low_pass_cutoff}
    
    # V. Convert back to torch tensor for downstream processing
    ## We have to expand the dimension and then convert to torch tensor since we are back to dealing with torchaudio
    audio_wav = torch.from_numpy(np.expand_dims(audio_wav, axis=0))
    
    return (audio_wav, sr, paras)

def echo_generator (wav_data, sr, delay, decay):
    # Initialise wave file
    delay_samples = int(delay * sr)
    echo_data = np.zeros_like(wav_data)
    
    # cater for situations where delay is longer than the audio itself
    # this will otherwise create an indexing error
    if delay_samples < len(wav_data) :
        echo_data[delay_samples:] = wav_data[:-delay_samples]*decay

    return (echo_data)