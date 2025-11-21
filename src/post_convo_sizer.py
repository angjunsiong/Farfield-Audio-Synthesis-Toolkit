## AJS's Afterthought - The Post-Convolution Sizer
## If you are naive like AJS you will think that you can just simply convolve an audio with a IR
## and then pass it on to the next stage of processs

## It is not so simple. There is a need fix the size of array after convolving:

## (1) When performing a room_IR or mobile convolution, you have to take into the account the
## time needed for the sound to travel from the source to the microphone
## There will be a delay between the original audio and the convolved audio
## because the convolved audio takes into account this time to travel.
## If we do not account for this delay, the original and convolved clip will
## have a misalignment, and it lead to poor results when using the
## clean-convolved pair for ML training 
## In practice, we will need to locate the first (and highest bump) on the IR,
## Then deduct away all audio samples within this time gap [t=0, t=1st_bump]
## If the convolved audio becomes shorter than the original audio after this deduction
## We will zero-pad the tail end of the convolved audio

## (2) Secondly, for fabric convolving, we will end up with an
## audio data file of size len(original audio) + len(convolution) - 1
## To ensure that the size of the original audio = size of convolved audio
## (which is essential for using the clean-convolved pair train a ML model)
## We will truncate away the tail-end of the convovled audio
## in contrast to room or mobile IR: the peak of fabric and HP IR happens virtually instantaneously
## (Fabric recording for e.g. at ~0.3m / mach 1 * 16kHz = 16 samples lag) + further mitigated by time-alignment upfront
## For a meeting room where sound travels about 10m, this would have been a ~500 sample lag 
## Thus there isn't a need to realign the peak

## Reflection: Gen AI certainly helped me this time. 
## I was asking chatGPT on the mechanics of room IR convolutions
## and it shun bian hinted me that I must account for the initial lag, if I am to use the
## clean-convolved pair to train a ML model 

import numpy as np
import torch


def post_convo_sizer(audio_data,
                     size_orig,  # Fed in from previous synthesis step
                     convo_type,  # "room", "mobile", or "fabric",
                     IR_applied=None):
    """
    Arguments:
    - torch tensor  audio_data  : The convolved audio data, to be correct-sized
    - int           size_orig   : The size of the original audio, before convolution
    - str           convo_type  : The type of convolution performed on the audio
                                : Either "room", "mobile", or "fabric" 
    - numpy_array   IR_applied  : The IR that was convolved onto the audio

    Returns:
    - torch tensor of dimension [len(adjusted_audio_data), 1]
    """
    ## Numpify torch tensor for following operations
    audio_data = np.squeeze(audio_data.numpy())

    ## For Room IR and mobile IR, detect initial peak in IR and deduct the time gap from front of convolved audio
    if convo_type == "room" or convo_type == "mobile":

        # Detect 1st (and logically highest peak in the room IR)
        peak_index = np.argmax(np.abs(IR_applied))
        print(f"IR peak detected at sample #{peak_index}")

        # Truncate front-end of convolved audio
        audio_data = audio_data[peak_index:]

        # If audio_data ends up being longer than the orginal audio_data, crop tail end
        if len(audio_data) >= size_orig:
            audio_data = audio_data[:size_orig]

        # if audio_data ends up being shorter than original audio, pad with zeros
        else:
            audio_data = np.pad(audio_data,
                                (0, size_orig - len(audio_data)),
                                mode="constant",
                                constant_values=0)

    ## Return audio to original size directly if convolving with fabric IR
    elif convo_type == "fabric":
        audio_data = audio_data[:size_orig]

    else:
        raise ValueError("please input a correct convo_type")

    ## Repack numpy array into a torch tensor
    audio_data = torch.from_numpy(np.expand_dims(audio_data, axis=0))

    return audio_data
