import random
import os
import tempfile
import json
from datetime import datetime

from src.audio_stacker import audio_noise_stack
from src.audio_effects_new import audio_effector
from src.helper_functions import *
from src.noise_builder import noise_builder
from src.noise_sizer import noise_sizer
from src.post_convo_sizer import post_convo_sizer
from src.ir_convolve import ir_convolve
from src.encoding_scripts.opus import encode_opus, decode_opus

def bulk_generation(number_of_audios = 10,
                    speech_folder = "./data/00_raw_speech/",
                    room_ir_folder = "./data/Impulse_Responses/room_IRs/",
                    noise_stationary_folder = "./data/01_stationary_noise/", 
                    noise_nonstationary_folder = "./data/02_non-stationary_noise/",
                    fabric_ir_folder ="./data/Impulse_Responses/fabric_IRs/",
                    handphone_ir_folder = "./data/Impulse_Responses/handphone_IRs/"):

    # Set up experiment log (for reproducibility)
    experiment_log = []
    
    for i in range(number_of_audios):

        # (Re)set up parameters log for audio
        parameters_log = {"serial": i,
                          "file_name": None, 
                          "original_speech_file":None,
                          "sampling_rate":None,
                          "sample_len":None,
                          "generate_clean_speech":None,
                          "add_room_reverb":None,
                          "stationary_noise":None,
                          "nonstationary_noise":None,
                          "combine_speech_noise":None,
                          "simulate_fabric":None,
                          "simulate_mobile":None,
                          "simulate_codec":None,
                          "phone_lowpass":None}

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
                        src = sample_data, 
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
        sample_data = post_convo_sizer(audio_data= sample_data, 
                                       size_orig = size_orig, 
                                       convo_type="room",
                                       IR_applied=IR_applied)
        
        # Log III-B Parameters:
        parameters_log["add_room_reverb"] = paras
        torchaudio.save("test_postroom.wav", sample_data, 16000, encoding="PCM_S", bits_per_sample=16)        
        
        ## Stage III-C: Synthesising Noise
        noise_stationary_data, sr, noise_stationary_paras = noise_builder(sample_data,
                                                            noise_stationary_folder, 
                                                            echo = True,
                                                            no_of_audio=random.randint(1,2), 
                                                            low_pass = True,
                                                            mode = "stationary")
        noise_nonstationary_data, sr, noise_nonstationary_paras = noise_builder(sample_data,
                                                                  noise_nonstationary_folder,
                                                                  no_of_audio=random.randint(0,2),
                                                                  echo = True,
                                                                  mode = "non-stationary") 

        # Log III-C Parameters:
        parameters_log["stationary_noise"] = noise_stationary_paras
        parameters_log["nonstationary_noise"] = noise_nonstationary_paras
        
        ## Stage III-D: Combining Speech and Noise
        stationary_nonstationary_NNR = random.uniform(-5,20)
        speech_noise_SNR  = random.uniform(-5,20)
        
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
                                                  "speech_noise_SNR": speech_noise_SNR}        

        torchaudio.save("test_postnoise.wav", sample_data, 16000, encoding="PCM_S", bits_per_sample=16)        


        ## Stage III-E: Simulating Passing of Audio through Fabric
        # 90% chance of mixing IRs, 10% chance of single random IR
        mode = random.choice(["random_mix"]*9+["random_single"]*1)
        sample_data, sr, size_orig, IR_applied, paras = ir_convolve(sample_data, 
                                                                    sr,
                                                                    mode=mode, 
                                                                    ir_repo=fabric_ir_folder)
        
        sample_data = post_convo_sizer(audio_data=sample_data, 
                                       size_orig=size_orig, 
                                       convo_type="fabric",
                                       IR_applied=IR_applied)

        # Log III-E Parameters:
        parameters_log["simulate_fabric"] = paras

        torchaudio.save("test_postfabric.wav", sample_data, 16000, encoding="PCM_S", bits_per_sample=16)        

        ## Stage III-F: Simulating Recording of Audio by Mobile Phones
        sample_data, sr, size_orig, IR_applied, paras = ir_convolve(sample_data, 
                                                             sr,
                                                             mode="random_mix",
                                                             ir_repo=handphone_ir_folder)
        
        sample_data = post_convo_sizer(audio_data=sample_data, 
                                       size_orig=size_orig, 
                                       convo_type="mobile",
                                       IR_applied=IR_applied)
        torchaudio.save("test_postmobile.wav", sample_data, 16000, encoding="PCM_S", bits_per_sample=16)        

        # Log III-F Parameters:
        parameters_log["simulate_mobile"] = paras
        
        ## Stage III-G. Simulating Degradation of Audio from Mobile CODEC Encoding/Decoding
        with tempfile.TemporaryDirectory() as tmpdirname:
            temp_file_path = os.path.join(tmpdirname, "sample_audio.wav")
            torchaudio.save(temp_file_path, sample_data, sample_rate=sr, encoding="PCM_S", bits_per_sample=16)
            ## Encode and Decode audio
            opus_encoded_path = encode_opus(wav_path = temp_file_path,
                                            tmp_folder=tmpdirname)
            opus_decoded_path = decode_opus(opus_encoded_path=opus_encoded_path,
                                            output_folder="./output/dirty_samples", count=str(i))
            
            print(f"audio {opus_decoded_path} generated!")

            # log parameters: file name
            parameters_log["file_name"] = opus_decoded_path.split('/')[-1]
            parameters_log["simulate_codec"] = "opus"

        # Append parameters to experiment log
        experiment_log.append(parameters_log)

    # Export parameters log as json
    # Use datetime module to serialise log file
    now = datetime.now()
    timestamp = now.strftime("%y%m%d_%H%M%S")
        
    with open(os.path.join("./output", f"experiment_log_{timestamp}.json"), "w") as f:
        json.dump(experiment_log, f, indent=2)

    return None