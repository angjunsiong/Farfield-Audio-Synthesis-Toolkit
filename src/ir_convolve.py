# AJS 18-mins Impulse Response Convolver
# This function convolves audio data with a IR fIle, to output an audio that has been shaped by an acoustic environment

# This was orginally developed for the Far-field Audio Synthesis Toolkit (FAST)
# This funciton is applied to convolve audio (say speech data) with:
# (1) Room IRs: acosustic information due to reverb and adsorption/attenuation over room distance
# (2) Fabric IRs: acosutic information when passing audio through different fabrics
# (3) HP Mic IRs: acoustic information when capturing audio with mobile phones 
                                                                                 
import numpy as np
from scipy import signal
import random
import torch
import os

def ir_convolve(audio_data,
                sr,
                mode = "random_mix", #random_mix, random_single, specific_mix, or specific
                ir_repo = None,
                no_of_ir = 4, # 10C4 for fabric, 18C4 for mobile: Ensure richness of IR samples
                mix_ir_list = None,
                specific_ir_path = None):
    """
    Arguments:
    - torch tensor  audio_data  : The audio data to be convolved
    - int       sr              : This is the sampling rate of audio_data
    - str       mode            : This indicates the mode used to pick out the ir to convolve audio data with
                                : (1) "random_mix" picks out no_of_ir from ir_repo and averages them
                                : (2) "single" picks out 1 ir from ir_repo
                                : (3) "specific_mix" uses the mean of IRs from a list of IRs
                                : This was created for regeneration purposes
                                : Can be used even with just 1 ir in list
                                : (4) "specific" uses the 1 x ir indicated in ir_path
    - str       ir_repo         : For "random_mix" and "random_single" modes, this is the repo to draw the random IRs from
    - int       no_of_ir        : For "random_mix" mode, indicates the number of IRs to draw from ir_repo
    - list      mix_ir_list     : For "specific_mix" mode, indicates IRs to be drawn from ir_repo
    - str       specific_ir_path: For "specific mode, indicates the path of the ir to be used

    Returns:
    - wav_data (torch tensor), sampling_rate (int), size of original audio (int), parameters (dict)
    """
    ## Set up parameters log
    paras = {}

    # Check audio: Check sampling rates (this function is built to use 16kHz IRs)
    if sr !=16000:
        raise ValueError ("Your Sampling Rate is not 16000kHz, which is what the IRs were built on.")

    ## I. Calculate IRs based on the diferent modes
    # Log parameters
    paras["mode"] = mode
    paras["RIRs_used"]=[]

    if mode == "random_mix":
        if not isinstance (no_of_ir, int):
            raise ValueError("Please indicate a valid no_of_ir (use integers)")
        ## Choose no_of_irs in ir_repo and average them
        chosen_ir= np.zeros_like(np.load(os.path.join(ir_repo, random.choice(os.listdir(ir_repo)))))
        
        for i in range(no_of_ir): # ??? is this ok
            sampled_ir = random.choice(os.listdir(ir_repo))
            # Log parameters
            paras["RIRs_used"].append(sampled_ir)

            chosen_ir += np.load(os.path.join(ir_repo, sampled_ir))
        
        chosen_ir = chosen_ir / no_of_ir
            
    elif mode == "random_single":
        sampled_ir = random.choice(os.listdir(ir_repo))
        chosen_ir = np.load(os.path.join(ir_repo, sampled_ir))
        
        # Log parameters
        paras["RIRs_used"].append(sampled_ir)

    elif mode == "specific_mix":
        for i in range(len(mix_ir_list)):
            if i == 0:
                mix_ir = np.load(os.path.join(ir_repo, mix_ir_list[i]))
            else:
                mix_ir = mix_ir + np.load(os.path.join(ir_repo, mix_ir_list[i]))
        # Average out ir
        chosen_ir = mix_ir / len(mix_ir_list)
        #print(chosen_ir)

    elif mode == "specific":
        chosen_ir = np.load(specific_ir_path)
        
        # Log parameters
        paras["RIRs_used"].append(chosen_ir)

    else:
        raise ValueError("Please indicate a valid mode: 'random_mix', 'random_single', 'specific_mix', or 'specific'")

    ## II. Convolve audio with ir
    # Only use full to capture every bit of IR details
    size_orig = len(torch.squeeze(audio_data).numpy())
    convolved_audio_data = signal.convolve(torch.squeeze(audio_data).numpy(), chosen_ir, mode="full")

    # Normalise data as it will become much softer
    max_value = np.max(np.abs(convolved_audio_data))
    if max_value > 0:
        convolved_audio_data = convolved_audio_data/max_value

    ## Repack into audio_data format as per pytorch
    convolved_audio_data = torch.from_numpy(convolved_audio_data)
    
    return convolved_audio_data, sr, size_orig, chosen_ir, paras