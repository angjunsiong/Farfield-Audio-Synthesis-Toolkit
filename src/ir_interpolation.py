"""
gemini generated this code to interpolate RIRs
TODO: check if this is the behavior we are looking for
probably works fine on fabric irs, and will mess up room irs
should work on microphone freq response type of irs too

also interpolating an ir with itself rewrites the phase, which might help with artifacts
"""

import numpy as np
import scipy.signal
import scipy.fft

def get_robust_peak_index(ir):
    """
    Finds the peak using the Hilbert Envelope.
    This works better for 'smeared' IRs where the energy is spread out
    and the highest single sample might not represent the center of energy.
    """
    # 1. Calculate Analytic Signal (Complex envelope)
    analytic_signal = scipy.signal.hilbert(ir)
    
    # 2. Get Amplitude Envelope
    envelope = np.abs(analytic_signal)
    
    # 3. Optional: Slight smoothing to ignore micro-transients
    # (Window size 5 is usually enough to smooth digital noise)
    envelope = scipy.signal.savgol_filter(envelope, 11, 3)
    
    return np.argmax(envelope)

def magnitude_to_minimum_phase_ir(magnitude_spectrum, n_fft):
    """
    Reconstructs a time-domain IR from a Magnitude Spectrum
    enforcing a Minimum Phase relationship (causal, max energy at start).
    """
    # 1. Log Magnitude
    # Add tiny epsilon to prevent log(0)
    log_mag = np.log(magnitude_spectrum + 1e-10)
    
    # 2. Inverse FFT to get Real Cepstrum
    cepstrum = np.fft.irfft(log_mag, n=n_fft)
    
    # 3. Window the Cepstrum to force causality
    # Minimum phase = Causal Cepstrum
    n_cep = len(cepstrum)
    window = np.zeros(n_cep)
    
    # Keep DC, Double positive frequencies, Zero negative frequencies
    window[0] = 1.0
    window[1 : n_cep//2] = 2.0
    window[n_cep//2] = 1.0 # Nyquist
    
    complex_cepstrum = cepstrum * window
    
    # 4. FFT back to frequency domain to get the Complex Spectrum
    # Exponentiate to undo the log
    min_phase_spectrum = np.exp(np.fft.rfft(complex_cepstrum, n=n_fft))
    
    # 5. Final Inverse FFT to time domain
    min_phase_ir = np.fft.irfft(min_phase_spectrum, n=n_fft)
    
    return min_phase_ir

def interpolate_irs_robust(ir1, ir2, alpha=0.5):
    """
    Robustly morphs between two IRs using Alignment + Min Phase Interpolation.
    
    Args:
        ir1, ir2: 1D numpy arrays (Audio data)
        alpha: Float 0.0 to 1.0 (Mix ratio. 0.0 = ir1, 1.0 = ir2)
    """
    # Ensure lengths match (pad to longest)
    max_len = max(len(ir1), len(ir2))
    # Round up to next power of 2 for FFT speed
    n_fft = 2**int(np.ceil(np.log2(max_len)))
    
    ir1_padded = np.pad(ir1, (0, n_fft - len(ir1)))
    ir2_padded = np.pad(ir2, (0, n_fft - len(ir2)))

    # --- STEP 1: ROBUST ALIGNMENT ---
    peak1 = get_robust_peak_index(ir1_padded)
    peak2 = get_robust_peak_index(ir2_padded)
    
    # Shift IR2 to match IR1
    shift = peak1 - peak2
    ir2_aligned = np.roll(ir2_padded, shift)
    
    # --- STEP 2: SPECTRAL INTERPOLATION ---
    # Get Magnitudes only (discard phase)
    mag1 = np.abs(np.fft.rfft(ir1_padded, n=n_fft))
    mag2 = np.abs(np.fft.rfft(ir2_aligned, n=n_fft))
    
    # Linear Interpolation of Magnitude
    # (This blends the 'EQ Curve' of the two sounds)
    mag_interp = (mag1 * (1 - alpha)) + (mag2 * alpha)
    
    # --- STEP 3: MINIMUM PHASE RECONSTRUCTION ---
    # This implicitly handles the 'smearing'. 
    # If the resulting magnitude implies a 'slow' filter (low pass),
    # the minimum phase reconstruction will create the correct time envelope.
    final_ir = magnitude_to_minimum_phase_ir(mag_interp, n_fft)
    
    # Normalize output (optional, to prevent clipping)
    # max_val = np.max(np.abs(final_ir))
    # if max_val > 0:
    #    final_ir = final_ir / max_val
        
    return final_ir

# --- Usage Example ---
if __name__ == "__main__":
    # Create two dummy IRs (one sharp, one smeared/delayed)
    t = np.linspace(0, 1, 16000)
    
    # Sharp spike at sample 100
    ir_sharp = np.zeros_like(t)
    ir_sharp[100] = 1.0 
    
    # Smeared bump at sample 200
    ir_smeared = np.exp(-((t - 0.02)**2) * 10000) # Gaussian bump
    
    # Morph!
    ir_morphed = interpolate_irs_robust(ir_sharp, ir_smeared, alpha=0.5)
    
    print(f"Generated IR length: {len(ir_morphed)}")
    print("The morphed IR will have a peak roughly at sample 0 (Min Phase property)")