## File export portion
## To do: Generate 10,000 IRs
## To do: Documentation

import argparse
import numpy as np
import matplotlib.pyplot as plt
import random
import pyroomacoustics as pra
import os # To do up file export portion
import time
import yaml

def rir_generate(quantity,
                 config_file,
                 master_seed,
                 output_folder,
                 preview_mode,
                 ):
    
    print(f"Welcome to the Amazing RIR Generator! You have ordered {quantity} RIRs!...")
    
    ## Extract parameters from config file
    with open(config_file, 'r') as file:
        parameters = yaml.safe_load(file)

    # (1) Room Dimensions
    base_length_upper, base_length_lower = parameters["room_dimensions"]["base_length_upper"], parameters["room_dimensions"]["base_length_lower"]
    base_width_upper, base_width_lower = parameters["room_dimensions"]["base_width_upper"], parameters["room_dimensions"]["base_width_lower"]
    entrance_depth_upper, entrance_depth_lower = parameters["room_dimensions"]["entrance_depth_upper"], parameters["room_dimensions"]["entrance_depth_lower"]
    entrance_length_upper, entrance_length_lower = parameters["room_dimensions"]["entrance_length_upper"], parameters["room_dimensions"]["entrance_length_lower"]
    radius_curve_upper, radius_curve_lower = parameters["room_dimensions"]["radius_curve_upper"], parameters["room_dimensions"]["radius_curve_lower"]
    height_upper, height_lower = parameters["room_dimensions"]["height_upper"], parameters["room_dimensions"]["height_lower"]
    curved_panels = 8
    margin = parameters["room_dimensions"]["margin"]

    # (2) Materials
    ceiling_mat = parameters["room_materials"]["ceiling"]
    floor_mat = parameters["room_materials"]["floor"]
    curved_walls_mat = parameters["room_materials"]["curved_walls"]
    entrance_mat = parameters["room_materials"]["entrance"]
    curtain_walls_mat = parameters["room_materials"]["curtain_wall"]
    scattering_mat = parameters["room_materials"]["scattering"]

    hard_scattering_upper, hard_scattering_lower = parameters["room_materials"]["scattering_values"]["hard_scattering_upper"], parameters["room_materials"]["scattering_values"]["hard_scattering_lower"]
    soft_scattering_upper, soft_scattering_lower = parameters["room_materials"]["scattering_values"]["soft_scattering_upper"], parameters["room_materials"]["scattering_values"]["soft_scattering_lower"]

    # (3) source and mic (including polar pattern values) parameters
    source_height_upper, source_height_lower = parameters["source"]["source_height_upper"],parameters["source"]["source_height_lower"] 
    mic_height_upper, mic_height_lower = parameters["mic"]["mic_height_upper"],parameters["mic"]["mic_height_lower"]
    mic_p_upper, mic_p_lower = parameters["mic"]["mic_p_upper"],parameters["mic"]["mic_p_lower"]

    # Start Timer
    start_time = time.perf_counter()

    for loop in range(quantity):

        # Set seed value
        # We set a seed number for each loop beginning with the master seed number
        # this is to facilitate the resumption of RIR generation in the event when something crashes
        # we will count the number of files in output folder and then add to the master seed each round
        # TODO: this makes the function non-idempotent, which is likely a bug
        # consider using the hash of a file as its seed instead
        seed = master_seed + len(os.listdir(output_folder))
        np.random.seed(seed)

        ### A. Create Room

        ## Randomise room dimensions
        base_length = np.random.uniform(base_length_upper, base_length_lower)
        base_width = np.random.uniform(base_width_upper, base_width_lower)
        entrance_depth = np.random.uniform(entrance_depth_upper, entrance_depth_lower)
        entrance_length = np.random.uniform(entrance_length_upper, entrance_length_lower)
        radius_curve = np.random.uniform(radius_curve_upper, radius_curve_lower)
        height = np.random.uniform(height_upper, height_lower)

        width_outside_curve = (base_width - (radius_curve*2))/2
        length_outside_entrance = (base_length - entrance_length)/2

        # A-I. Left Wall
        coordinate_1 = [radius_curve,0]
        coordinate_2 = [radius_curve,width_outside_curve]
        # start curve side 1
        coordinate_3 = [np.round(radius_curve-radius_curve*np.sin(np.pi/curved_panels), decimals=2), np.round(radius_curve - radius_curve*np.cos(np.pi/curved_panels) + width_outside_curve, decimals=2)]
        coordinate_4 = [np.round(radius_curve-radius_curve*np.sin(2*np.pi/curved_panels), decimals=2), np.round(radius_curve - radius_curve*np.cos(2*np.pi/curved_panels) + width_outside_curve, decimals=2)]
        coordinate_5 = [np.round(radius_curve-radius_curve*np.sin(3*np.pi/curved_panels), decimals=2), np.round(radius_curve - radius_curve*np.cos(3*np.pi/curved_panels) + width_outside_curve, decimals=2)]
        coordinate_6 = [np.round(radius_curve-radius_curve*np.sin(4*np.pi/curved_panels), decimals=2), np.round(radius_curve - radius_curve*np.cos(4*np.pi/curved_panels) + width_outside_curve, decimals=2)]
        coordinate_7 = [np.round(radius_curve-radius_curve*np.sin(5*np.pi/curved_panels), decimals=2), np.round(radius_curve - radius_curve*np.cos(5*np.pi/curved_panels) + width_outside_curve, decimals=2)]
        coordinate_8 = [np.round(radius_curve-radius_curve*np.sin(6*np.pi/curved_panels), decimals=2), np.round(radius_curve - radius_curve*np.cos(6*np.pi/curved_panels) + width_outside_curve, decimals=2)]
        coordinate_9 = [np.round(radius_curve-radius_curve*np.sin(7*np.pi/curved_panels), decimals=2), np.round(radius_curve - radius_curve*np.cos(7*np.pi/curved_panels) + width_outside_curve, decimals=2)]
        coordinate_10 = [np.round(radius_curve-radius_curve*np.sin(8*np.pi/curved_panels), decimals=2), np.round(radius_curve - radius_curve*np.cos(8*np.pi/curved_panels) + width_outside_curve, decimals=2)]
        # end curve side 1
        coordinate_11 = [radius_curve, base_width]

        # A-II. top wall (entrance)
        coordinate_12 = [radius_curve+length_outside_entrance, base_width]
        coordinate_13 = [radius_curve+length_outside_entrance, base_width + entrance_depth]
        coordinate_14 = [radius_curve+length_outside_entrance + entrance_length, base_width + entrance_depth]
        coordinate_15 = [radius_curve+length_outside_entrance + entrance_length, base_width]
        coordinate_16 = [radius_curve+base_length, base_width]

        ## A-III. Right Wall
        # start curve side 2
        coordinate_17 = [radius_curve + base_length, base_width - width_outside_curve]
        coordinate_18 = [radius_curve + base_length + np.round(radius_curve*np.sin(7*np.pi/curved_panels), decimals=2), np.round(radius_curve - radius_curve*np.cos(7*np.pi/curved_panels) + width_outside_curve, decimals=2)]
        coordinate_19 = [radius_curve + base_length + np.round(radius_curve*np.sin(6*np.pi/curved_panels), decimals=2), np.round(radius_curve - radius_curve*np.cos(6*np.pi/curved_panels) + width_outside_curve, decimals=2)]
        coordinate_20 = [radius_curve + base_length + np.round(radius_curve*np.sin(5*np.pi/curved_panels), decimals=2), np.round(radius_curve - radius_curve*np.cos(5*np.pi/curved_panels) + width_outside_curve, decimals=2)]
        coordinate_21 = [radius_curve + base_length + np.round(radius_curve*np.sin(4*np.pi/curved_panels), decimals=2), np.round(radius_curve - radius_curve*np.cos(4*np.pi/curved_panels) + width_outside_curve, decimals=2)]
        coordinate_22 = [radius_curve + base_length + np.round(radius_curve*np.sin(3*np.pi/curved_panels), decimals=2), np.round(radius_curve - radius_curve*np.cos(3*np.pi/curved_panels) + width_outside_curve, decimals=2)]
        coordinate_23 = [radius_curve + base_length + np.round(radius_curve*np.sin(2*np.pi/curved_panels), decimals=2), np.round(radius_curve - radius_curve*np.cos(2*np.pi/curved_panels) + width_outside_curve, decimals=2)]
        coordinate_24 = [radius_curve + base_length + np.round(radius_curve*np.sin(1*np.pi/curved_panels), decimals=2), np.round(radius_curve - radius_curve*np.cos(1*np.pi/curved_panels) + width_outside_curve, decimals=2)]

        ## A-IV. Bottom Wall
        coordinate_25 = [radius_curve + base_length, width_outside_curve]
        coordinate_26 = [radius_curve + base_length, 0]

        
        
        ### B. Choose materials; prepare material list
        scattering_mat_chosen = random.choice(scattering_mat)
        
        # B-I: walls
        curved_walls_mat_0to11_13to24 = pra.Material(random.choice(curved_walls_mat), scattering = scattering_mat_chosen)
        entrance_mat_12 = pra.Material(random.choice(entrance_mat), scattering = np.random.uniform(hard_scattering_upper, hard_scattering_lower)) # hard surface
        curtain_walls_mat_25 = pra.Material(random.choice(curtain_walls_mat), scattering = np.random.uniform(soft_scattering_upper, soft_scattering_lower)) # soft surface
        wall_materials = [curved_walls_mat_0to11_13to24 for i in range(12)] + [entrance_mat_12] + [curved_walls_mat_0to11_13to24 for i in range(12)] + [curtain_walls_mat_25]
        
        # B-II: ceiling and floor
        ceiling_mat_27 = pra.Material(random.choice(ceiling_mat))
        ceiling_scattering = np.random.uniform(hard_scattering_upper, hard_scattering_lower) # hard surface
        floor_mat_26 = pra.Material(random.choice(floor_mat))
        floor_scattering = np.random.uniform(soft_scattering_upper, soft_scattering_lower) # soft surface

        
        
        ### C.Create room in pyroomacoustics
        # C-I: Create 2-D room
        corners = np.array([coordinate_1, coordinate_2, coordinate_3, coordinate_4, coordinate_5, 
                    coordinate_6, coordinate_7, coordinate_8, coordinate_9, coordinate_10,
                    coordinate_11, coordinate_12, coordinate_13, coordinate_14, coordinate_15,
                    coordinate_16, coordinate_17, coordinate_18, coordinate_19, coordinate_20,
                    coordinate_21, coordinate_22, coordinate_23, coordinate_24, coordinate_25,
                    coordinate_26]).T  # [x,y]
        
        room = pra.Room.from_corners(corners, 
                                     fs = parameters["acoustics"]["sampling_rate"],
                                     materials = wall_materials,
                                     max_order = parameters["acoustics"]["rir_simulation_order"])

        # C-II. Extrude into 3-D rooms
        room.extrude(height = height)
        room.walls[-2].absorption = floor_mat_26.absorption_coeffs.copy() # floor
        room.walls[-2].scatter = [ceiling_scattering] # floor 
        room.walls[-1].absorption = ceiling_mat_27.absorption_coeffs.copy() # ceiling
        room.walls[-1].scatter = [floor_scattering]# ceiling
        
        # Inspect materials
        # for count, i in enumerate(room.walls):
        #     print(count, i.absorption, i.scatter)

        ### D. Set up Source and Mic position
        # source
        source_height = np.random.uniform(source_height_lower, source_height_upper)
        source_x = radius_curve + np.random.uniform(margin*base_length, base_length*(1-margin))
        source_y = np.random.uniform(margin*base_length, base_width*(1-margin))
        
        room.add_source([source_x, source_y, source_height])

        # mic position
        mic_height = np.random.uniform(mic_height_lower, mic_height_upper)
        mic_x = radius_curve + np.random.uniform(margin * base_length, base_length*(1-margin))
        mic_y = np.random.uniform(margin * base_width, base_width*(1-margin))
        mic_pos = np.array([[mic_x, mic_y, mic_height]]).T

        # mic polar pattern and mic directivity
        mic_p = np.random.uniform(mic_p_lower, mic_p_upper)
        azimuth = np.random.uniform(0, 2 * np.pi)
        colatitude = np.random.uniform(0, np.pi)

        directivity = pra.directivities.CardioidFamily(
            orientation=pra.directivities.DirectionVector(azimuth=azimuth, colatitude=colatitude, degrees=False),  # (azimuth, colatitude) in rads
            p = mic_p
            )
        
        my_mic = pra.MicrophoneArray(mic_pos, fs=parameters["acoustics"]["sampling_rate"], directivity=directivity)
        room.add_microphone_array(my_mic)

        if preview_mode:
            print(f"Source position: {source_x}, {source_y}, {source_height}")
            print(f"Mic position: {mic_x}, {mic_y}, {mic_height}")

            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d') # default to first grid position (row, column, index)
            room.plot(ax=ax)   # now room is 3-D
            plt.title("3-D Extruded Room")
            plt.show()

        ### E. Simulate and Generate RIR
        # Run Simulation;
        room.compute_rir()
        rir = room.rir[0][0]

        # Step 1: Generate and generated RIR of a moderate order up to first 0.2s
        rir_simulated_duration_upper = parameters["acoustics"]["rir_simulated_duration_upper"]
        rir_simulated_duration_lower = parameters["acoustics"]["rir_simulated_duration_lower"]
        rir_simulated_duration = np.random.uniform(rir_simulated_duration_lower, rir_simulated_duration_upper)
        rir_simulated_truncated_length = int(rir_simulated_duration * room.fs)
        rir_simulated_truncated = rir[:rir_simulated_truncated_length]

        # Step 2: Approximate long tail of RIR using exponential decay (i.e. total 0.6 seconds)
        rir_total_duration_upper = parameters["acoustics"]["rir_total_duration_upper"]
        rir_total_duration_lower = parameters["acoustics"]["rir_total_duration_lower"]
        rir_total_duration = np.random.uniform(rir_total_duration_lower, rir_total_duration_upper) # Also known as T60
        
        tail_length = int(rir_total_duration * room.fs - rir_simulated_truncated_length)
        time_tail = np.arange(tail_length) / room.fs
        late_tail = np.abs(rir_simulated_truncated[-1]) * np.exp(-6.91 * (time_tail + rir_simulated_duration/room.fs) / rir_total_duration) # note scaling
        
        # Step 3: Concatenante RIR segments
        rir_final = np.concatenate([rir_simulated_truncated, late_tail])

        ##############

        if preview_mode:
            # Time axis for rir_full
            t_full = np.arange(len(rir)) / room.fs
            # Time axis for rir_final
            t_final = np.arange(len(rir_final)) / room.fs

            # Plot
            plt.figure(figsize=(10,5))
            plt.plot(t_full, rir, label='Full RIR with moderate order', alpha=0.8, linewidth = 0.7)
            plt.plot(t_final, rir_final, label='Early + Late Tail', alpha=0.8)
            plt.xlabel('Time [s]')
            plt.ylabel('Amplitude [dB]')
            plt.title('Comparison of Full RIR vs Simulated Moderate-Order Head + Exp Decay Tail')
            plt.grid(True)
            plt.legend()
            plt.show()

        ### F. Export RIR; Must to prevent overwriting what have been generated (in case this was after a crash)
        rir_filename = f"rir_{len(os.listdir(output_folder))}.npy"
        if not os.path.isdir(output_folder):
            os.mkdir(output_folder)
        np.save(os.path.join(output_folder, rir_filename), rir_final)

        if preview_mode:
            print("Preview Mode is On... Generating only 1 RIR!")
            break

        if (loop+1)%10 == 0:
            end_time = time.perf_counter()
            elapsed_time = (end_time - start_time) / 60
            print(f"{loop+1} out of {quantity} RIR generated... Elapsed time: {elapsed_time:.1f} minutes")

    return (None)

if __name__ == "__main__":    
    parser = argparse.ArgumentParser(description="Your RIR Generator")
    parser.add_argument("--quantity", type=int, default=500, help="Number of RIRs to Generate")
    parser.add_argument("--config_file", type=str, default = "./config.yaml", help="The config yaml file to refer to")
    parser.add_argument("--master_seed", type=int, default = 42, help="Set initial seed")
    parser.add_argument("--output_folder", type=str, default = "./simulated_RIRs", help="Set output folder")
    parser.add_argument("--preview_mode", type = bool, default=False, help="Generate only 1 RIR, albeit with illustrative schematics if True")

    args = parser.parse_args()

    rir_generate(args.quantity,
                 args.config_file,
                 args.master_seed,
                 args.output_folder,
                 args.preview_mode)
