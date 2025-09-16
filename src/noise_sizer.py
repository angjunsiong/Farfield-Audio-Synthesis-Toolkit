## AJS's Magnificent Audio Padder • Truncetor
# This function randomly pads some zeros, to prepore a noise before overlaying a noise over an audio elip
# The intent is to randomise the insertion of a non-stationary noise into training data to enrich them
# This was developed as part of the Far-Field Audio Synthesis Toolkit (FAST)

# A thing to look out for here lies in ensuring that we do not cause indexing errors because the
# a noise clip is longer than the audio to be overlaid, thereby creating indexing issues downstream

import numpy as np
import random
import torch
import torch.nn.functional as F

def noise_sizer(audio_data_1,
                audio_data_2, 
                mode = "stationary",
                pad_size = None): #"stationary" or "non-stationary"
    """
    Ensures that the length of noise data (audio_data_2) matches that of audio data (audio_data_1) by
    (A) for stationary noises: truncate or loop noise
    (B) for non-stationary noises: adding random pads to leading and/or trailing edge of audio data (and truncate if necessary)

    Arguments:
    - torch_tensor audio_data : This 1s the audio data whose size will be taken reference against (1.0. output nust natch this size)
    - torch_tensor noise_data : This is the audio data whose size we are adjusting to natch the size of audio_data,1
    - str mode      : This indicates if we are trying to insert a "stationary" or "non-stationary" noise
                    : (A) If "stationary" (developed for insertion of stationary noise)
                    : - We will truncate the noise (audio_dato_2) if it is longer than the reference (audio_data_1)
                    : - We will loop the noise (audio_data_2) if it is shorter than the reference (audio_data_1)
                    : (B) If "non-stationary" (developed for insertion of non stationary noise)
                    : - We will insert a (random quantity) of leading zeros to noise (audio_data _2)
                    : - Then we will either (1) pad the trailing end of audio_data_2 if the padded audio_data_2 is still shorter than the reference (audio_data_1)
                    : - (ii) or truncate audio_data_2 1f padded audio_dato_2 becomes longer then audio_data_1 after padding
    - int pad_size  : For "non-stationary" mode, this gives the zero-padding ahead of introducing the nonstationary noise
    
    Returns:
    - torch tensor of dimension (1, n_samples), sampling_rate (int)

    """
    ## SCENARIO 1: Stationary Mode
    if mode == "stationary":
    
        # I. We consider what is the size of len(audio_data_1) relative to len(audio dato_2)
        size_of_1_over_2 = len(audio_data_1[0]) / len(audio_data_2[0])

        # II-A: If size_of_1_over_2 < 1; i.e. audio_data_2 is longer than audio_data_1
        ## We will truncate audio_data_2 to match size of audio_data_1
        if size_of_1_over_2 < 1:
            looped_audio_data_2 = audio_data_2[:,:len(audio_data_1[0])]

        # II-B: If size_of_1_over_2 > 1; i.e. audio_data_2 is shorter than audio_data_1
        ## We will loop the audio_data_2 ceiling (size_of_1_over_2) times, 
        ## then truncate it to match the size of audio_data_1
        else:
            looped_audio_data_2 = audio_data_2.repeat(1,(int(size_of_1_over_2)+1))
            looped_audio_data_2 = looped_audio_data_2[:,:len(audio_data_1[0])]

        return looped_audio_data_2, pad_size

    ## SCENARIO 2: Non-Stationary Mode
    if mode == "non-stationary":
        # I. Pad noise_data with random number of leading zeros; 
        # random number is between 0 and the length of audio_data
        ## generate random pad size
        # In practice, a None pad_size only happens when we are doing bulk generation
        # It should be a determined figure when performing audio regeneration
        if pad_size is None:
            pad_size = random.randint(0, audio_data_1.shape[1])
        ## add trailing zeros
        padding_front = (pad_size,0)
        padded_audio_data_2 = F.pad(audio_data_2, pad=padding_front, mode="constant", value=0)

        # II. Adjust size of noise-data to match audio data
        ## If noise data ends up longer than audio data, truncate
        if len (padded_audio_data_2[0]) > len(audio_data_1[0]):
            padded_audio_data_2 = padded_audio_data_2[:, :len(audio_data_1[0])]
        ## If padded_noise_data is shorter, then top up with trailing zeros
        else:
            padding_rear = (0, len(audio_data_1[0]) - len(padded_audio_data_2[0]))
            padded_audio_data_2 = F.pad(padded_audio_data_2, pad = padding_rear, mode="constant", value=0)
        
        return padded_audio_data_2, pad_size