import random
import os
import tempfile
import json

from src.audio_stacker import audio_noise_stack
from src.audio_effects_new import audio_effector
from src.helper_functions import *
from src.noise_builder import noise_builder
from src.noise_sizer import noise_sizer
from src.post_convo_sizer import post_convo_sizer
from src.ir_convolve import ir_convolve
from src.encoding_scripts.opus import encode_opus, decode_opus
from src.phone_lowpass import phone_augment

def regenerate_dataset(log_json,
                       speech_folder = "./data/00_raw_speech/",
                       room_ir_folder = "./data/Impulse_Responses/room_IRs/",
                       # TODO: warning! you're not using these input, instead you're hardcoding some values below
                       # noise_stationary_folder = "./data/01_stationary_noise/",
                       # noise_nonstationary_folder = "./data/02_non-stationary_noise/",
                       fabric_ir_folder ="./data/Impulse_Responses/fabric_IRs/",
                       handphone_ir_folder = "./data/Impulse_Responses/handphone_IRs/"):

    # Read json file
    with open(log_json, "r") as f:
        log = json.load(f)

    print(f"{len(log)} audio files to be regenerated...")

    # Loop through log, log down parameters
    for audio_serial in range(len(log)):
        
        ## III-A. Generate clean speech
        # Retrieve required parameters
        file_name = log[audio_serial]["file_name"]
        original_speech_file = log[audio_serial]["original_speech_file"]
        tempo_change_rate = log[audio_serial]["generate_clean_speech"]["tempo_change_rate"]
        pitch_shift = log[audio_serial]["generate_clean_speech"]["pitch_shift"]
        
        # File check
        if not os.path.isfile(os.path.join(speech_folder, original_speech_file)):
            print (f"{os.path.join(speech_folder, original_speech_file)} not found")
            print (f"Skipping regen of {file_name}")
            continue

        # Load original audio file
        sample_data, sr = load_audio_with_pytorch(os.path.join(speech_folder, log[audio_serial]["original_speech_file"]))
        # Implement efects on speech data (only tempo and pitch shift)
        sample_data, sr, _ = audio_effector(sample_data,
                                                tempo_change=True,
                                                tempo_range=(tempo_change_rate, tempo_change_rate),
                                                pitch_shift=True,
                                                pitch_shift_range=(pitch_shift, pitch_shift))
        
        ## III-B: Synthesising Speech with Room Reverberation
        # Retrieve required parameters
        room_ir = log[audio_serial]["add_room_reverb"]["RIRs_used"][0]
        
        # File check        
        if not os.path.isfile(os.path.join(room_ir_folder, room_ir)):
            print (f"{os.path.join(room_ir_folder, room_ir)} not found")
            print (f"Skipping regen of {file_name}")
            continue

        # Convolve data with random room IR
        sample_data, sr, size_orig, IR_applied, _ = ir_convolve(sample_data, sr,
                                                                mode="specific",
                                                                specific_ir_path=os.path.join(room_ir_folder, room_ir))

        # Rightsize convolved data
        sample_data = post_convo_sizer(audio_data= sample_data, 
                                       size_orig = size_orig, 
                                       convo_type="room",
                                       IR_applied=IR_applied) 

        
        ## III-C: Synthesising Noise
        
        # Retrieve required parameters
        stationary_paras = log[audio_serial]["stationary_noise"]
        nonstationary_paras = log[audio_serial]["nonstationary_noise"]

        ### Part 1: Rebuild stationary noise
        stationary_noise_list = []

        # a. Build base noise stack of a zero-array; return if # stationary noise is 0 
        noise_stationary_data = torch.zeros_like(sample_data) + 1e-14

        # b. Regenerate piecewise noise
        for stationary_serial in range(len(stationary_paras)):

            noise_count = stationary_serial+1
            noise_name = stationary_paras[f"noise_{noise_count}"]["noise_name"]
            
            ## Pick out required parameters
            # We only pick out effect variables after verifying that effects is not empty
            # Else it will lead to an indexing error
            if len(stationary_paras[f"noise_{noise_count}"]["effects"]) != 0:
                # This check is required because the number of echos might be 0
                # In this case, we make sure echo is set to False
                if "list_of_delays" in stationary_paras[f"noise_{noise_count}"]["effects"]:
                    list_of_delays = stationary_paras[f"noise_{noise_count}"]["effects"]["list_of_delays"]
                    list_of_decays = stationary_paras[f"noise_{noise_count}"]["effects"]["list_of_decays"]
                    echo = True
                else:
                    list_of_delays = None
                    list_of_decays = None
                    echo = False

                low_pass_order = stationary_paras[f"noise_{noise_count}"]["effects"]["low_pass"]["low_pass_order"]
                low_pass_cutoff = stationary_paras[f"noise_{noise_count}"]["effects"]["low_pass"]["low_pass_cutoff"]
            
            NNR = stationary_paras[f"noise_{noise_count}"]["noise_to_stack_NNR"]
            pad_size = stationary_paras[f"noise_{noise_count}"]["pad_size"]

            # Read audio file
            noise_path = os.path.join("./data/01_stationary_noise", noise_name)
            noise_data, _ = load_audio_with_pytorch(noise_path)
            
            # Rebuild noise
            noise_data, _ , _ = audio_effector(audio_wav=noise_data, 
                                                     sr = sr,
                                                     echo = echo,
                                                     low_pass = True,
                                                     list_of_delays = list_of_delays,
                                                     list_of_decays = list_of_decays,
                                                     low_pass_order = (low_pass_order, low_pass_order),
                                                     low_pass_cutoff = (low_pass_cutoff, low_pass_cutoff)
                                                     )
            
            # Size data
            noise_data, _ = noise_sizer(sample_data, noise_data, mode = "stationary")
            
            # Append to noise_data
            stationary_noise_list.append(noise_data)

            # Stack noise using NNR data
            # For first loop, set noise_data as noise_stack_data 
            if noise_count == 1:
                noise_stationary_data = noise_data

            # For second loop and above, add noise_data to noise_stack_data at prescribe NNR
            else:
                noise_stationary_data = F.add_noise(noise_stationary_data, noise_data, torch.tensor([NNR]))
        
            # Note that in the event where number of noise = 0, 
            # the zero array noise_stack_data will be passed on to the next stage of code

        ### Part 2: Rebuild nonstationary noise
        nonstationary_noise_list = []

        # a. Build base noise stack of a zero-array; return this if # nonstationary noise is 0 
        noise_nonstationary_data = torch.zeros_like(sample_data) + 1e-14

        # b. Regenerate piecewise noise
        for nonstationary_serial in range(len(nonstationary_paras)):

            noise_count = nonstationary_serial+1
            noise_name = nonstationary_paras[f"noise_{noise_count}"]["noise_name"]
            
            ## Pick out required parameters
            # We only pick out effect variables after verifying that effects is not empty
            # Else it will lead to an indexing error
            if len(nonstationary_paras[f"noise_{noise_count}"]["effects"]) != 0:
                # This check is required because the number of echos might be 0
                # In this case, we make sure echo is set to False
                if "list_of_delays" in nonstationary_paras[f"noise_{noise_count}"]["effects"]:
                    list_of_delays = nonstationary_paras[f"noise_{noise_count}"]["effects"]["list_of_delays"]
                    list_of_decays = nonstationary_paras[f"noise_{noise_count}"]["effects"]["list_of_decays"]
                    echo = True
                else:
                    list_of_delays = None
                    list_of_delays = None  # TODO: i think you meant `list_of_decays`, likely a bug
                    echo = False
            
            NNR = nonstationary_paras[f"noise_{noise_count}"]["noise_to_stack_NNR"]
            pad_size = nonstationary_paras[f"noise_{noise_count}"]["pad_size"]

            # Read audio file
            noise_path = os.path.join("./data/02_non-stationary_noise", noise_name)
            noise_data, _ = load_audio_with_pytorch(noise_path)
            
            # Rebuild noise
            noise_data, _ , _ = audio_effector(audio_wav = noise_data, 
                                               sr = sr,
                                               echo = echo,
                                               list_of_delays = list_of_delays,
                                               list_of_decays = list_of_decays
                                               )
            
            # Size data
            noise_data, _ = noise_sizer(sample_data, noise_data, mode = "non-stationary", pad_size=pad_size)
            
            # Append to noise_data
            nonstationary_noise_list.append(noise_data)

            # Stack noise using NNR data
            # For first loop, set noise_data as noise_stack_data 
            if noise_count == 1:
                noise_nonstationary_data = noise_data

            # For second loop and above, add noise_data to noise_stack_data at prescribe NNR
            else:
                noise_nonstationary_data = F.add_noise(noise_nonstationary_data, noise_data, torch.tensor([NNR]))
        
            # Note that in the event where number of noise = 0, 
            # the zero array noise_stack_data will be passed on to the next stage of code


        ## III-D: Combining Speech and Noise

        # Retrieve parameters
        stationary_nonstationary_NNR = log[audio_serial]["combine_speech_noise"]["stationary_nonstationary_NNR"]
        speech_noise_SNR = log[audio_serial]["combine_speech_noise"]["speech_noise_SNR"]

        # Combine noise and clean speech using retrieved parameters

        combined_noise_data = audio_noise_stack(noise_stationary_data, 
                                                noise_nonstationary_data,
                                                stationary_nonstationary_NNR
                                                )

        sample_data = audio_noise_stack(sample_data, 
                                        combined_noise_data,
                                        speech_noise_SNR
                                        )
        

        ## III-E: Simulating Passing of Audio through Fabric
        if log[audio_serial]["simulate_fabric"] is not None:
            # Retrieve required parameters
            fabric_irs = log[audio_serial]["simulate_fabric"]["RIRs_used"]
            
            # File check
            skip_iter = False
            for fabric_ir in fabric_irs:
                if not os.path.isfile(os.path.join(fabric_ir_folder, fabric_ir)):
                    print (f"{os.path.join(fabric_ir_folder, fabric_ir)} not found")
                    skip_iter = True
            if skip_iter == True:
                print (f"Skipping regen of {file_name}")
                continue

            # Convolve data with fabric IR
            sample_data, sr, size_orig, IR_applied, _ = ir_convolve(sample_data, 
                                                                    sr,
                                                                    ir_repo=fabric_ir_folder,
                                                                    mode="specific_mix",
                                                                    mix_ir_list=fabric_irs)

            # Rightsize convolved data
            sample_data = post_convo_sizer(audio_data= sample_data, 
                                        size_orig = size_orig, 
                                        convo_type="fabric",
                                        IR_applied=IR_applied)

        ## III-F: Simulating Recording of Audio by Mobile Phones
        
        if log[audio_serial]["simulate_mobile"] is not None:
            # Retrieve required parameters
            mobile_ir = log[audio_serial]["simulate_mobile"]["RIRs_used"]
            
            # File check
            skip_iter = False
            
            for ir in mobile_ir:
                if not os.path.isfile(os.path.join(handphone_ir_folder, ir)):
                    print (f"{os.path.join(handphone_ir_folder, ir)} not found")
                    skip_iter = True
            if skip_iter == True:
                print (f"Skipping regen of {file_name}")
                continue

            # Convolve data with mobile IR
            sample_data, sr, size_orig, IR_applied, _ = ir_convolve(sample_data, 
                                                                    sr,
                                                                    ir_repo=handphone_ir_folder,
                                                                    mode="specific_mix",
                                                                    mix_ir_list=mobile_ir)

            # Rightsize convolved data
            sample_data = post_convo_sizer(audio_data= sample_data, 
                                        size_orig = size_orig, 
                                        convo_type="mobile",
                                        IR_applied=IR_applied)

        ## III-G: Simulating Recording of Audio by Mobile Phones
        if log[audio_serial]["simulate_codec"] is not None: 
            # Retrieve required parameters
            codec = log[audio_serial]["simulate_codec"]
            file_name= log[audio_serial]["file_name"]

            if codec == "opus":
                with tempfile.TemporaryDirectory() as tmpdirname:
                    temp_file_path = os.path.join(tmpdirname, "sample_audio.wav")
                    torchaudio.save(temp_file_path, sample_data, sample_rate=sr, encoding="PCM_S", bits_per_sample=16)
                    ## Encode and Decode audio
                    opus_encoded_path = encode_opus(wav_path = temp_file_path,
                                                    tmp_folder=tmpdirname)
                    opus_decoded_path = decode_opus(opus_encoded_path=opus_encoded_path,
                                                    output_folder="./output/regenerated_samples",
                                                    count=str(audio_serial),
                                                    decoded_path=file_name)            

            else:
                print("codec not supported")
                continue

        ## III: Simulating phone with simple bandpass filter
        # Note this is mutually exclusive with III-E,F,G
        if log[audio_serial]["phone_lowpass"] is not None:
            sample_data, sr = phone_augment(sample_data, sr)
            # Export file
            torchaudio.save(os.path.join("./output/regenerated_samples",f"phone_lowpass_sample_{audio_serial}.wav"), 
                            sample_data, sr, encoding="PCM_S", bits_per_sample=16)

        if (audio_serial+1)%100==0: 
            print(f"{audio_serial+1} files generated!")

    print("Regeneration Complete!")

    return log