import subprocess
import os

def encode_opus(wav_path, tmp_folder, bit_rate = "24k", sampling_rate = 16000):
    # Use ffmpeg to encode WAV file to Opus
    encoded_path = os.path.join(tmp_folder, wav_path.replace('.wav','.opus'))
    # control bitrate with "-b:a 96k"; sampling_rate with "-ar 48000"
    # echo y to answer "y" to overwrite by default
    command = f"echo y | ffmpeg -i {wav_path} -c:a libopus -b:a {bit_rate} -ar 16000 -nostats -hide_banner -loglevel error {encoded_path}"
    subprocess.run(command, shell=True, check=True)

    return encoded_path

def decode_opus(opus_encoded_path, output_folder, encoding = "pcm_s16le", sampling_rate = 16000, count="", decoded_path = None):
    # Use ffmpeg to decode the Opus file back to WAV
    if decoded_path == None:
        decoded_path = os.path.join(output_folder, count+"_"+opus_encoded_path.split('/')[-1].replace('.opus','_opus_decoded.wav'))
    else:
        decoded_path = os.path.join(output_folder, decoded_path)
    # control encoding "-c:a pcm_s16le"; sampling_rate with "-ar 48000"
    # echo y to overwrite by default
    command = f"echo y | ffmpeg -i {opus_encoded_path} -c:a {encoding} -ar {sampling_rate} -nostats -hide_banner -loglevel error {decoded_path}"
    subprocess.run(command, shell=True, check=True)
    
    return decoded_path

def main():
    input_wav_path = 'input.wav'
    
    # Encode the WAV file to Opus
    encoded_path = encode_opus(input_wav_path)
    print(f"Encoded to: {encoded_path}")
    
    # Decode the Opus encoded audio back to WAV
    decoded_path = decode_opus(encoded_path)
    print(f"Decoded to: {decoded_path}")
    
if __name__ == "__main__":
    main() 