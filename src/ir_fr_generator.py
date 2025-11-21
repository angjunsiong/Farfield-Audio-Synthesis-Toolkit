# AJS men vibe-coded Impulse Response Generator # 6
# This function calculates the impulse (time domain) response given 2 sine sweeps 
# i.e. before & after passing through a medium, such as a fabric
# Note to self: This was actually 88% vibe-coded!


#### This should probably be in the same py as ir_convolve
# Purpose of this function? Isn't it mathematically equivalent to just multiplying by room RIR?

import os

import librosa
import matplotlib.pyplot as plt
import numpy as np
from scipy.fft import fft
from scipy.fft import fftfreq
from scipy.signal import savgol_filter
from scipy.signal.windows import tukey


def impulse_generator(
        reference_path='./data/fabric_experiment/aligned_run2_recordings/ref_trimmed_0Clean_0deg_NoCover_aligned.wav',
        input_folder='./data/fabric_experiment/aligned_run2_recordings/',
        output_folder='./data/Impulse_Responses/fabric_IRs/',
        target_sr=16000,
        truncate_imp_resp=True,
        suppress_low_freq_noise=True,
        visualise_freq_response=False,
        visualise_imp_response=False,
        just_preview_1=False):
    """
    Argument:
    - str   reference_path  : The path to the audio (in wav) before being altered from being passed thror
                            : This is defaulted to "ref_trinmed_OClean_Odeg_aligned.wav" - the reference
    - str   input_folder    : The folder containing the audio-s (in wav) after passing through the fabric
    - str   output_folder   : The folder which the calculated IRS will be exported to
    - int   targer_sr       : The sampling of the impulse response to be generated
                            : This must be compatible with the audio that you are planning to convolv wi
                            : In project FAST, the IR is calculated using audio of 48kHz, but will be re:
    - bool suppress_imp_resp_spike : If True, will remove spikes at ends of impulse response due to downsampling
    - bool suppress_low_freq_noise : If True, will set calculated freq response between 0 and 100Hz to 0db
                                   : This considers the noisiness at that frequency band
                                   : and instability at the edges when performing FT
    - bool visualise_freq_response : If True, will show the freq repsonse plot generated from analysis
    - bool visualise_imp_response : If True, will show the impulse repsonse plot generated from analysis
    - bool just_preview_1   : If True, will only run function for 1 impulse response (for troubleshootin
    
    Return: None

For info; fabrics used in experiments
1PT: SAF PT Pants
2GW: SAF Garrison Wear Pants
3BP: Business Pants
4LF: Long 4
5HD: Hoodie
6LS: Long Sleeve Shirt
7SW: Fleece Sweater
8CT: Cyberthon Bag
9LC: Longchamp Bag
10BJ: Blue Jeans
"""

    print("GENERATING IMPULSE!!\n")
    print(f"{len(os.listdir(input_folder))} files detected in input folder {input_folder}")

    ## Load reference audio; Apply full window and perform FULL fourier transform
    ref_sweep, sr = librosa.load(reference_path, sr=None)
    # Implement window - use a hanning or tukey window because it provides (Tukey is essentially hanning with steeper slopes)
    # low alpha value is recommended for noisy signal to preserve energy of the signal ???
    # When alpha = 0, window is rectangular; When alpha = 1, window is hanning
    # The steep slopes will preserve energy magnitudes better, while avoiding sharp transitions ("wraparound effect")
    # (1) Good sharpness reduction/transition properties - ultimately what we want to reduce wrap around and spectral leakages
    # (2) symmetric - since low and high frequencies equally important???
    # (3) strong main lobe and low side lobes allow good balance of time resolution and frequency resolution
    # window = get_window('hann', len(ref_sweep),)
    window = tukey(len(ref_sweep), alpha=0.05)
    ref_sweep = ref_sweep * window
    # apply FFT with a smallest power of 2
    fft_N = 2 ** int(np.ceil(np.log2(len(ref_sweep))))
    fft_ref = fft(ref_sweep, fft_N)

    ## Normalise audio
    ### Why normalise?
    ### (1) Prevents clipping of loud noise to cause distortion
    ### (2) Standardise input against imperfect experimental technique
    ### (3) Little downstream issues since we are more interested in relative response among frequencies ???
    fft_ref = librosa.util.normalize(fft_ref)

    ## Create folder if not already there
    if not os.path.isdir(output_folder):
        os.mkdir(output_folder)

    ## Loop through sine sweep folder and generate impulse response
    list_of_files = os.listdir(input_folder)
    list_of_files.sort()

    for file in list_of_files:
        # Exclude the reference file itself from being processed
        if file != reference_path.split("/")[-1]:

            # I. Load Calculated Impulse Response File
            output_sweep, sr_sweep = librosa.load(os.path.join(input_folder, file), sr=None)
            # Apply same tukey window to output_sweep
            output_sweep = output_sweep * window
            fft_output = fft(output_sweep, fft_N)
            fft_output = librosa.util.normalize(fft_output)

            # II. FREQ-DOMAIN Noise Management measure 1: IMPLEMENT THRESHOLD
            ##  Calculates magnitude of points on the spectrum and implement threshold
            ## This is a very blunt tool and we should not apply a big threshold factor here
            magnitudes_output = np.abs(fft_output)
            threshold_output = 0.000001 * np.max(magnitudes_output)
            magnitudes_ref = np.abs(fft_ref)
            threshold_ref = 0.000001 * np.max(magnitudes_ref)

            ## Apply filter
            filtered_fft_output = np.copy(fft_output)
            filtered_fft_output[magnitudes_output < threshold_output] = 0

            filtered_fft_ref = np.copy(fft_ref)
            filtered_fft_ref[magnitudes_ref < threshold_ref] = 0

            # III. ??? FREQ-DOMAIN Noise Management measure 2 >> Use Wiener Cross Spectral Density + Tikhonov Regularisation
            # We apply (1) Wiener Cross Spectral Density (CSD)+ (2) Tikhonov Regularisation approach
            # (1) Wiener Cross Spectral Density: Measures correlation of 2 signals IN THE FREQ DOMAIN
            # "cross spectral density = A FT of a convolution between 2 signals"
            # "transforms time-doamin cross-correlation function into the frequency domain"

            # Layman explanation:
            # "If a frequency is strong in the original signal and strongly correlated in the recorded signal"
            # "Then it is probably a true part of the medium's effect"
            # "If it doesn't match well, then it is probably just noise"

            # Effectively separate noise from actual signals
            # Numerator S_xy: Multiply fft_recorded and conjugate of FFT_reference to get cross-correlation
            # Denominator S_xx: The magnitude (or power of the reference signal) for normalisation 

            # Technique provides a robust IR in the presence of noise
            # This is a big step up from naive division: freq_resp = fft_output/(fft_ref + 1e-12)
            # It particularly avoid spikes due to division of a very low number, typically caused by noise
            # i.e. small change in observed data can lead to large change in reconstructed signal

            # (2) Tikhonov method introduces a "penalty for solutions that are too noisy or oscillate wildly"
            # Thisis also Ridge Regularisation > the same techqniue to prevent coefficients from growing too wildly
            # higher lambda = more smoothed out response curve
            # !!!: This math minimises the Mean Squared Error between estimated original signal and actual original signal

            # Calculate Wiener Regularised (with Tikhonov regularisation) frequency response:
            lambda_reg = 1e-3  # arbitrary >> Low lambda: more details but amplifies noise; high lambda: more noise suppresison at expense of details???
            freq_resp_raw = filtered_fft_output * np.conj(filtered_fft_ref) / (
                        np.abs(filtered_fft_ref) ** 2 + lambda_reg)

            # IV. Keep relevant frequencies (up to target frequencies)
            # Retain frequency axis up to nysquist of target frequency
            nyquist_target = target_sr / 2
            # Realign frequency axis: Compute length of original time domain, and determine size of each frequency b
            N = len(freq_resp_raw) * 2 - 1
            freqs = np.fft.rfftfreq(N, d=1 / sr_sweep)
            # Keep only frequency components under nyquist_target
            freq_resp_downsampled = freq_resp_raw[freqs <= nyquist_target]
            freqs_downsampled = freqs[freqs <= nyquist_target]
            N_downsampled = len(freqs_downsampled)

            # IV. FREQ-DOMAIN Noise Management measure 3: Implement spectral smoothing
            # Create smoothed verison for freq response
            # Savitzky-Golay Filter: Tries to fit a polynomial of certain order over window_length, and reduce least squared error
            # One of most widely cited paper in Analytical Chemistry
            # # We use a savgol filter - favoured in signal processing because it closely adapts to audio / sinusoid signals
            # To note, we can only apply the savgol filter on the magnitude of the FFT, not on the complex values
            # so we will need to reconstruct the impulse response using filtered magnitude and the original phase data
            filter_window_size = 521
            # Apply savgol fi l t e r on mag of freq_resp;mag_db_freq_resp_cleaned is used for visualisation
            mag_db_freq_resp_downsampled = 20 * np.log10(np.abs(freq_resp_downsampled) + 1e-12)
            mag_db_freq_resp_cleaned = savgol_filter(mag_db_freq_resp_downsampled,
                                                     window_length=filter_window_size,
                                                     polyorder=3)
            # reconstruct phase data
            phase = (np.angle(freq_resp_downsampled))
            ## Convert back to linear magnitude and attach back phase information
            freq_resp_cleaned = 10 ** (mag_db_freq_resp_cleaned / 20) * np.exp(1j * phase)

            # V. ???FREQ-DOMAIN Noise Management Measure 4: Supprese low frequencies
            # We note significont distortions with the frequency response at the 0 to 100kz range
            # We set the frequency response of the band 0 to 300kz to 0.5 gain on linear magnitude scole
            # but we are not using O directly to compensate for the drop in amplitude due to the fabric
            # This is calibrated using hearing tests (Where, at unity gain, low frequencies were observed to be too noisy
            # The impact is likely to be low since this ia the frequency band that will be muddled with noise in practice

            if suppress_low_freq_noise:
                freq_to_suppress = 400
                frequency_per_bin = target_sr / N_downsampled
                # TODO: did you mean `*=` (make it smaller) instead of `=` (also deletes all complex phase)
                # alternatively use a butter bandpass filter like in the workshop
                freq_resp_cleaned[0:int(np.ceil(freq_to_suppress / frequency_per_bin))] = 0.45

            # V. Perform inverse-FFT (and take real part) to get impulse response (in time domain)
            impulse_response_cleaned = np.real(np.fft.ifft(freq_resp_cleaned))

            # A technique that didn't work well ???
            # RM VI. Resample obtained freq_resp and impulse response to target sample rate
            # RM impulse_response_smooth_resampled = resample_poly(impulse_response_smooth, up=1, down=3)

            # VI. TIME-DOMAIN Noise Management: Impulse Response (IR) Filter:
            filter_window_size = 21
            impulse_response_cleaned_smoothed = savgol_filter(impulse_response_cleaned,
                                                              window_length=filter_window_size,
                                                              polyorder=4)

            ## Optional: Suppress spike at end of impulse response
            # This is to remove spike due to downsampling effect
            # We can remove large chunk of it since the most critical part is the initial impulse
            # (i.e. will not require 5 whole sec of impulse, esp when the trailing end is prone to distortions like ringing
            # in practice, unlikely for reverbs and time-shifts to last 5 seconds
            if truncate_imp_resp:
                samples_to_keep = 600  # about ~0.3s
                impulse_response_cleaned = impulse_response_cleaned[:samples_to_keep,]
                impulse_response_cleaned_smoothed = impulse_response_cleaned_smoothed[:samples_to_keep,]

            ### Normalisation: Not necessary to implement here

            ## Normalise by peak amplitude
            # impulse_response_cleaned_norm = impulse_response_cleaned /пр. max (n.abs (impulse_response_cleaned))
            # impulse_response_cleaned_smoothed_norm = impulse_response_cleaned_smoothed / np.max(np. abs(impulse_resi

            ## Alternately, normalise impulse response by energy
            # Mathematically, it scales the impulse response such that energy to 1, but it just reduces the energy ti
            # impulse_response_cleaned = impulse_response_cleaned / np. Linalg.norm(impulse_response_cleaned)
            # impulse_response_cleaned_smoothed = impulse_response_cleaned_smoothed / np. linalg.norm(impulse_responsi

            # Note: everything after this is just for visualisation purposes

            # For visualisation

            if visualise_freq_response:
                ## Fit axes into new sampling rate
                freqs = fftfreq(N_downsampled, d=1 / target_sr)
                pos_freqs = freqs[:N_downsampled // 2]
                H_half_raw = freq_resp_downsampled[:N_downsampled // 2]
                H_half_cleaned = freq_resp_cleaned[:N_downsampled // 2]

                ## convert to db scale
                mag_raw = 20 * np.log10(np.abs(H_half_raw) + 1e-10)
                mag_cleaned = 20 * np.log10(np.abs(H_half_cleaned) + 1e-10)
                plt.figure(figsize=(5, 2))
                plt.axhline(y=8, color="r")
                plt.plot(pos_freqs, mag_raw, linewidth=0.3, color="orange")
                plt.plot(pos_freqs, mag_cleaned, linewidth=1, color="blue")
                plt.ylim(-50, 20)
                plt.xlabel("Frequency")
                plt.ylabel("Amplitude in dB")
                plt.title(f"freq resp: {file}")

            if visualise_imp_response:
                # Visualise impulse response
                # Smoothen inpulse response curve
                plt.figure(figsize=(5, 2))
                plt.plot(impulse_response_cleaned, linewidth=0.7, color='orange')
                # plt.plot(impulse_response_cleaned_norm, linewidth = 0.3, color = 'orange")
                plt.plot(impulse_response_cleaned_smoothed, linewidth=0.7, color='blue')
                # plt.plot(impulse_response_cleaned_smoothed_norm, linewidth = 0.7, color = 'blue')
                plt.title(f"Impulse Reponse: {file}")
                plt.xlabel("Samples")
                plt.ylabel("Amplitude")
                plt.grid(True)
                plt.show()

            # Export Smooth version
            # np.save(os.path.join(output_folder,f"ir_{file[:-4]}_smooth.npy"), impulse_response_cleaned_smoothed_norm
            np.save(os.path.join(output_folder, f"ir_{file[:-4]}_smooth.npy"), impulse_response_cleaned_smoothed)

        else:
            print(f"Skipping reference file {reference_path.split('/')[-1]}...")

        if just_preview_1:
            break

    print("IMPULSE COMPLETED! \n\n'But you don't need to use the claw when you pick a pear of the big pawpaw'")

    return (None)
