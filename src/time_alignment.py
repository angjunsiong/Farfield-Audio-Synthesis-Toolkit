## AJS's Super Audio Aligner
# This function time-aligns different audio clips recorded under different conditions
# In this implementation, a folder of audios are aligned against a reference audio, 
# and then export the aligned audios to a designated folder

# Use Case: This was originally developed for the Far-Field Audio Synthesis Toolkit (FAST)
# We have wanted to examine the frequency response when audio is recorded through a fabric
# So experiments were conducted to collect recordings of sine sweep-s when they are passed through certain fabrics
# However, because the recordings might start at different point of the clips, we will need to align them
# Conventionally, people uses a loud impulse (Like the "ACTION" during filming) to time align different recordings
# Since this is not done at the point of the experiment, we relied on spectrogram cross-corcelation to time-align

# For consistency, do note that we should deliberately introduce a lag to 
# the audio to be aligned against the reference audio
# i.e. The reference audio i s always "ahead"
# This can be done using a simple audio editing tool such as audacity
# This is a measure to prevent removing useful portions of the audios we wish to align

# Note to self: This was FUCKING cool when you finally got it to work!
# The absence of a ready package doing this already is so surprising!
# Perhaps publish a python package for it?

import os

import librosa
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf  # to export librosa arrays into wav
from scipy.signal import correlate


def time_aligner(reference_path="./data/fabric_experiment/references/0Clean_0deg_NoCover_aligned.wav",
                 input_folder="./data/fabric_experiment/run2_recording",
                 output_folder="./data/fabric_experiment/aligned_run2_recordings",
                 hop_length=32,
                 trim=True,
                 trim_duration_s=4.5,
                 preview_spec=False,
                 just_preview_1=False):
    """
    Time-aligns a folder of wav audio clips against a reference wav clip, and export aligned audio-s into 
    a designated folder.
    
    Arguments:
    - str   reference_path  : The path to the audio (in wav) which audios in input_folder will take alignment against
                            : This is defaulted to "OClean_Odeg_aligned.wav" - the reference (recording without passing through any fabric)
    - str   input_folder    : The folder containing the audio-s (in wav) to be time-aligned against the reference audio
    - str   output_folder   : The folder which the time-aligned audio-s will be exported to
    - int   hop_length      : hop size for the stft when representing the audios in spectrogram form
                            : pertinent when performing cross-correlation between the 2 spectrograms
                            : Defaulted to low value of 32 for higher time resolution; can increase to lower processing time
    - bool  trim            : if True, will truncate the aligned audio clip to duration as indicated in trim_durations_s.
    - float trim_duration_s : The duration which aligned audio clips will be truncated to;
                            : Defaulted to ~4.s since this is approximately where the sine sweep covers up to 16kHz
    - bool  preview_spec    : If True, will print the spectrogram slices for preview / troubleshooting purposes
    - bool  just_preview_1  : If True, will only align 1 x audio clip for preview / troubleshooting purposes

    return : NaN
    raises ValueError : if the sampling rates for both audio do not match

    To Note:
    - default stft frame size is 2048.    
    """

    # A. Read Reference audio (in this case will be the sine sweep recorded without passing through without any fabric)
    data_ref, sr1 = librosa.load(reference_path, sr=None)

    # B. Loop through audios to be aligned
    files_count = len([file for file in os.listdir(input_folder) if "wav" in file])
    print(f" {files_count} files found in folder {input_folder}\n")

    for count, audio in enumerate(os.listdir(input_folder)):
        if "wav" not in audio:
            continue

        print("-------------")
        print(f"Processing file {count + 1} of {files_count} files: {audio}")

        # C. Read in audio to be aligned
        data_fabric, sr2 = librosa.load(os.path.join(input_folder, audio), sr=None)

        # D. Check that sample rates are the same
        if sr1 == sr2:
            print(f"The sampling rates of the reference and the audio to be aligned are both {sr1}Hz!\n")
        else:
            raise ValueError("The sampling rates of the clips are not the same.")

        # E. Compute Spectrogram for both audio (magnitude)
        # default stft size if 2048
        spec_ref = np.abs(librosa.stft(data_ref, hop_length=hop_length))
        spec_audio = np.abs(librosa.stft(data_fabric, hop_length=hop_length))

        # F. Cross-correlate spectrograms
        ## F1: pad the shorter spectrogram to match the lengths between reference and audio to be aligned
        ## index [1] refers to the columns, or the time-axis
        if spec_ref.shape[1] > spec_audio.shape[1]:
            spec_audio = np.pad(spec_audio, ((0, 0), (0, spec_ref.shape[1] - spec_audio.shape[1])), mode="constant")
        else:
            spec_ref = np.pad(spec_ref, ((0, 0), (0, spec_audio.shape[1] - spec_ref.shape[1])), mode="constant")

        # F2: Removing low frequencies before time-aligning
        ## Remove spectrum from 0 to 2000Hz: Because this band is too noisy and can mess with alignment
        n_rows_to_remove = int(2000 / (8000 / 2048))  # calculate # of rows to remove, using freq band / freq resolution
        ## Remove rows
        spec_audio = spec_audio[:(len(spec_audio) - n_rows_to_remove)]
        spec_ref = spec_ref[:(len(spec_ref) - n_rows_to_remove)]

        # F3: Thresholding to reduce noise
        ## Arbitrary value of multiplier 2 used
        ## That is, any spectrogram value lower than 2 * the mean of the spectrogram values will be set to zero
        ## This will wipe out any low magnitude noises present inthe spectrogram
        arbitrary_noise_coefficient = 2
        spec_audio[spec_audio < np.mean(spec_audio) * arbitrary_noise_coefficient] = 0
        spec_ref[spec_ref < np.mean(spec_ref) * arbitrary_noise_coefficient] = 0

        # F4: Multiple points alignment
        ## we chop the spectrogram up into 5 pieces and average out the alignment timing
        spec_sub_arrays_ref, spec_sub_arrays_audio = np.array_split(spec_ref, 5, axis=0), np.array_split(spec_audio, 5,
                                                                                                         axis=0)
        list_of_time_lag_hops = []

        # We drop the last slice, since it doesn't lie in the frequeney band of interest to us
        # Recall that we are only interested up to 16kHz ???
        for index in range(len(spec_sub_arrays_ref) - 1):
            correlation = correlate(spec_sub_arrays_audio[index], spec_sub_arrays_ref[index],
                                    mode="full",
                                    method="fft")

            # sticking with full... I know it is a waste of resources since we are only looking for time-shift
            # (i.e. There isn't a need to perform corelation along the y-axis, since the time misalignment only occurs along time or x-axis)
            # Not sure how to only do a "full" correlation only along x-axis
            if preview_spec == True:
                print(f"\nPreview Spec Mode is: ON. Displaying spectrograms for slice {index}...")
                print(f"LHS: Reference, RHS: Audio to be Aligned")

                fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2)

                S_db = librosa.amplitude_to_db(np.abs(spec_sub_arrays_ref[index]), ref=np.max)
                img = librosa.display.specshow(S_db,
                                               sr=sr1,
                                               x_axis="s",
                                               y_axis="off",
                                               ax=ax1,
                                               hop_length=hop_length)

                S_db = librosa.amplitude_to_db(np.abs(spec_sub_arrays_audio[index]), ref=np.max)
                img = librosa.display.specshow(S_db,
                                               sr=sr1,
                                               x_axis="s",
                                               y_axis="off",
                                               ax=ax2,
                                               hop_length=hop_length)
                plt.show()

            # G. Find the peak in the correlation result
            ## unravel index indicates the index of a value (in this case the "shift" which gives the max correlation) inside the correlation matrix
            ## We first find the flat index of the max correlation value in the correlation matrix with np.argmax
            ## Then we find the row and column position of this flat index inside the correlation matrix (in terms of row column)
            ## example: if array shape is 3x3 (which we feed in as correlation.shape below)
            ## then a flat index of 7 will return the lag_indices (2,1) »> meaning row 2 and column 1
            ## The column number is what we are interested in. It corresponds to the tie-shift in terms of number of hops
            ## https://stackoverflow.com/questions/48135736/what-is-an-intuitive-explanation-of-np-unravel-index
            lag_indices = np.unravel_index(np.argmax(correlation), correlation.shape)
            time_lag_hops = lag_indices[1] - (spec_audio.shape[1] - 1)  # adjust for full mode

            ## using index [1] for lag indices means we are taking the column (which corresponds to time-shift)
            ## Note time lag per hop = hop_length / sr1
            list_of_time_lag_hops.append(time_lag_hops)
            print(
                f"Slice {index} Analysis: Clip {audio} lags behind reference by {time_lag_hops} hops, or {time_lag_hops * hop_length / sr1}s.")

        ## Reject any outliers in the list of lags; Happens when there is just too much noise about certain slices
        # ## Arbitrarily, reject time that is more than 1.2 x std deviation away from mean of all lags
        selection_boolean = list(
            abs(list_of_time_lag_hops - np.mean(list_of_time_lag_hops)) < 1.2 * (np.std(list_of_time_lag_hops) + 1e-4))
        print(f"Selection Boolean: {selection_boolean}")
        list_of_time_lag_hops = [i for index, i in enumerate(list_of_time_lag_hops) if selection_boolean[index] == True]

        # H. Finally, remove the lag to the audio and export the adjusted audio
        ## Designate output file name
        output_file_name = audio[:-4] + "_aligned.wav"

        ## Convert lag (currently measured in STFT hop frames) into number of samples to cut
        print("\n*************************")
        print(
            f"Average of Analyses: Clip {audio} lags behind reference by average of {np.mean(list_of_time_lag_hops) * hop_length / sr1}s")
        time_lag_seconds = np.mean(list_of_time_lag_hops) * hop_length / sr1
        samples_lag = int(time_lag_seconds * sr1)

        ## Align audio: This is achieved by removing the front part of the "lagging" audio
        if time_lag_hops >= 0:
            aligned_data_fabric = data_fabric[samples_lag:]  # removes empty front part
        else:
            print("WARNING: You are seemingly removing non-empty portion of the audio you are trying to align")
            # Message should not appear if we have already deliberately "lagged" the clip to be aligned during pre-pi
            aligned_data_fabric = data_fabric[samples_lag:]  # removes non-empty front part

        ## Trim audio: Now we trim the end of the aligned data_fabric
        if trim == True:
            aligned_data_fabric = aligned_data_fabric[:int(trim_duration_s * sr1)]

        ## Will trim reference too; This will just output the trimmed reference into the output folder
        path_name = (f"ref_trimmed_{reference_path.split('/')[-1]}")
        if not os.path.isfile(os.path.join(output_folder, path_name)):
            sf.write(os.path.join(output_folder, path_name), data_ref[:int(trim_duration_s * sr1)], samplerate=sr1)

        ## Export adjusted audio
        if not os.path.isdir(output_folder):
            os.mkdir(output_folder)

        sf.write(os.path.join(output_folder, output_file_name), aligned_data_fabric, samplerate=sr1)
        print(f" {output_file_name} exported!")
        print("************************\n\n")

        if just_preview_1:
            break
