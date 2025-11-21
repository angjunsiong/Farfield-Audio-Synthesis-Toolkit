#### Add some docstrings if this is not a util function
# Add rng seed
# Should probably allow for log output dir/name to be customised

import json
import random
from datetime import datetime

from src.audio_effects_new import audio_effector
from src.audio_stacker import audio_noise_stack
from src.utils.loader import load_audio_with_pytorch
from src.ir_convolve import ir_convolve
from src.noise_builder import noise_builder
from src.phone_lowpass import phone_augment
from src.post_convo_sizer import post_convo_sizer


def bulk_generation_simple(number_of_audios=10,
                           speech_folder="./data/00_raw_speech/",
                           room_ir_folder="./data/Impulse_Responses/room_IRs/",
                           noise_stationary_folder="./data/01_stationary_noise/",
                           noise_nonstationary_folder="./data/02_non-stationary_noise/"
                           ):
    # Set up experiment log (for reproducibility)
    experiment_log = []

    for i in range(number_of_audios):
        # (Re)set up parameters log for audio
        parameters_log = {"serial":                i,
                          "file_name":             None,
                          "original_speech_file":  None,
                          "sampling_rate":         None,
                          "sample_len":            None,
                          "generate_clean_speech": None,
                          "add_room_reverb":       None,
                          "stationary_noise":      None,
                          "nonstationary_noise":   None,
                          "combine_speech_noise":  None,
                          "simulate_fabric":       None,
                          "simulate_mobile":       None,
                          "simulate_codec":        None,
                          "phone_lowpass":         None}

        ## Stage III-A: Generate Clean Speech
        # Load random audio from raw speech folder
        speech_file = random.choice(os.listdir(speech_folder))
        sample_data, sr = load_audio_with_pytorch(os.path.join(speech_folder, speech_file))
        # Implement efects on speech data (only tempo and pitch shift)
        sample_data, sr, paras = audio_effector(sample_data,
                                                tempo_change=True,
                                                pitch_shift=True)

        # Clean speech generated
        torchaudio.save(f"./output/clean_samples/{i}.wav",
                        src=sample_data,
                        format="wav",
                        encoding="PCM_S",
                        sample_rate=sr,
                        bits_per_sample=16)

        # Log III-A Parameters:
        parameters_log["original_speech_file"] = speech_file
        parameters_log["sampling_rate"] = sr
        parameters_log["sample_len"] = sample_data.shape[1]
        parameters_log["generate_clean_speech"] = paras

        torchaudio.save("test_preroom.wav", sample_data, 16000, encoding="PCM_S", bits_per_sample=16)

        ## Stage III-B: Synthesising Speech with Room Reverberation 
        # Convolve data with random room IR
        sample_data, sr, size_orig, IR_applied, paras = ir_convolve(sample_data, sr,
                                                                    mode="random_single",
                                                                    ir_repo=room_ir_folder)

        # Rightsize convolved data
        sample_data = post_convo_sizer(audio_data=sample_data,
                                       size_orig=size_orig,
                                       convo_type="room",
                                       IR_applied=IR_applied)

        # Log III-B Parameters:
        parameters_log["add_room_reverb"] = paras
        torchaudio.save("test_postroom.wav", sample_data, 16000, encoding="PCM_S", bits_per_sample=16)

        ## Stage III-C: Synthesising Noise
        noise_stationary_data, sr, noise_stationary_paras = noise_builder(sample_data,
                                                                          noise_stationary_folder,
                                                                          echo=True,
                                                                          no_of_audio=random.randint(1, 2),
                                                                          low_pass=True,
                                                                          mode="stationary")
        noise_nonstationary_data, sr, noise_nonstationary_paras = noise_builder(sample_data,
                                                                                noise_nonstationary_folder,
                                                                                no_of_audio=random.randint(0, 2),
                                                                                echo=True,
                                                                                mode="non-stationary")

        # Log III-C Parameters:
        parameters_log["stationary_noise"] = noise_stationary_paras
        parameters_log["nonstationary_noise"] = noise_nonstationary_paras

        ## Stage III-D: Combining Speech and Noise
        stationary_nonstationary_NNR = random.uniform(-5, 20)
        speech_noise_SNR = random.uniform(-5, 20)

        combined_noise_data = audio_noise_stack(noise_stationary_data,
                                                noise_nonstationary_data,
                                                stationary_nonstationary_NNR
                                                )
        sample_data = audio_noise_stack(sample_data,
                                        combined_noise_data,
                                        speech_noise_SNR
                                        )

        # Log III-D Parameters:
        parameters_log["combine_speech_noise"] = {"stationary_nonstationary_NNR": stationary_nonstationary_NNR,
                                                  "speech_noise_SNR":             speech_noise_SNR}

        print(sample_data.shape)
        print(type(sample_data))

        torchaudio.save("test_postnoise.wav", sample_data, 16000, encoding="PCM_S", bits_per_sample=16)

        ## Apply Low-Pass Filter to Simulate Fabric, Mobile, and Mobile Codec Encoding/Decoding
        sample_data, sr = phone_augment(sample_data, sr)
        print(sample_data.shape)
        print(type(sample_data))

        # Log phone_lowpass parameters
        parameters_log["phone_lowpass"] = True

        # Export file
        torchaudio.save(os.path.join("./output/dirty_samples", f"phone_lowpass_sample_{i}.wav"),
                        sample_data, sr, encoding="PCM_S", bits_per_sample=16)

        # Append parameters to experiment log
        experiment_log.append(parameters_log)

    # Export parameters log as json
    # Use datetime module to serialise log file
    now = datetime.now()
    timestamp = now.strftime("%y%m%d_%H%M%S")

    with open(os.path.join("./output", f"experiment_log_{timestamp}.json"), "w") as f:
        json.dump(experiment_log, f, indent=2)

    return None
