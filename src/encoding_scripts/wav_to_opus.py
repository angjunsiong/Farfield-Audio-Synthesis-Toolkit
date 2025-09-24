import wave

from pyogg_encoder import OggOpusWriter
from pyogg_encoder import OpusBufferedEncoder


def wav_to_opus(wav_input_path: str, opus_output_path: str):

    # Open the input WAV file
    with wave.open(wav_input_path, 'rb') as wav_file:
        # Get audio parameters from the WAV file
        channels = wav_file.getnchannels()
        sampling_frequency = wav_file.getframerate()
        bytes_per_sample = wav_file.getsampwidth()

        # The Opus encoder requires 16-bit audio
        if bytes_per_sample != 2:
            print(f"Error: WAV file must be 16-bit, but this file is {bytes_per_sample * 8}-bit.")
            return

        print(f"WAV file properties: {channels} channels, {sampling_frequency} Hz")

        # Configure the Opus encoder
        encoder = OpusBufferedEncoder()
        encoder.set_application('audio')
        encoder.set_sampling_frequency(sampling_frequency)
        encoder.set_channels(channels)
        encoder.set_frame_size(20)

        # Initialize the OggOpus writer with the encoder
        writer = OggOpusWriter(opus_output_path, encoder)

        print(f"Encoding '{wav_input_path}' to '{opus_output_path}'...")

        # Read the WAV in chunks and feed it to the writer
        chunk_size_frames = 4096
        while True:
            pcm_chunk = wav_file.readframes(chunk_size_frames)
            if not pcm_chunk:
                break
            # The writer handles the buffering and encoding
            writer.write(memoryview(pcm_chunk))

    # IMPORTANT: Close the writer to finalize the file
    writer.close()

    print("Conversion successful.")


if __name__ == '__main__':
    wav_to_opus("sample1.wav", "sample1_converted.opus")
