import wave

from pyogg_encoder import OpusFileStream


def opus_to_wav(opus_input_path: str, wav_output_path: str):
    # Open the input Opus file
    opus_stream = OpusFileStream(opus_input_path)

    # Open the output WAV file
    with wave.open(wav_output_path, 'wb') as wav_file:
        wav_file.setnchannels(opus_stream.channels)
        wav_file.setsampwidth(2)  # for Opus files, the decoded output is always 16-bit PCM (2 bytes per sample)
        wav_file.setframerate(opus_stream.frequency)

        # Read and decode Opus data in chunks
        while True:
            buffer = opus_stream.get_buffer()
            if buffer is None:
                break

            wav_file.writeframes(buffer)


if __name__ == '__main__':
    opus_to_wav("sample1.opus", "sample1.wav")
