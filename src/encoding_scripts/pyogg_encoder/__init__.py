import os
import wave
from enum import Enum

from . import opus
from .ogg_opus_writer import OggOpusWriter
from .opus_buffered_encoder import OpusBufferedEncoder
from .opus_file_stream import OpusFileStream
from .pyogg_error import PyOggError


class OpusApplication(Enum):
    """
    An enumeration for the available Opus codec application modes.

    This controls the internal encoder optimizations.
    """
    # `AUDIO` is best for music or mixed content (prioritizes the CELT layer).
    # Gives the highest quality for non-speech audio.
    AUDIO = "audio"

    # `VOIP` is best for voice-only applications (prioritizes the SILK layer).
    # Optimized for speech intelligibility at lower bitrates.
    VOIP = "voip"

    # Low-latency mode for applications requiring minimal delay.
    RESTRICTED_LOWDELAY = "restricted_lowdelay"


def encode_opus(
        wav_path: str,
        tmp_folder: str,
        application: OpusApplication = OpusApplication.AUDIO,
        sampling_rate: int = 16000,
        frame_size: int = 20,
        bitrate: int | None = 128000,
        vbr: bool = True,
        complexity: int | None = 10
) -> str:
    """
    Encodes a WAV file to an OggOpus file using the vendored pyogg_encoder.

    This function provides direct access to libopus settings for higher quality
    encoding, allowing for explicit control over the SILK (speech) and CELT
    (music) layers of the Opus codec.

    :param wav_path: Path to the input WAV file. Must be 16-bit PCM.
    :param tmp_folder: Folder where the temporary output Opus file will be stored.
    :param application: The Opus application mode to use for encoding.
    :param sampling_rate: The sampling rate of the audio in Hz (e.g., 16000).
    :param frame_size: Opus frame size in milliseconds (e.g., 20, 40, 60).
    :param bitrate: Target bitrate in bits per second (e.g., 64000, 128000).
                    If None, the codec's default is used.
    :param vbr: If True, enables Variable Bitrate for better quality.
    :param complexity: Encoder complexity (0-10). 10 is highest quality but slowest.
                       If None, the codec's default is used.
    :return: The path to the encoded Opus file.

    :raises PyOggError: If an error occurs during the encoding process.
    """
    # Define the output path for the encoded opus file
    base_name = os.path.basename(wav_path)
    opus_name = os.path.splitext(base_name)[0] + '.opus'
    encoded_path = os.path.join(tmp_folder, opus_name)

    try:
        # Open the input WAV file to read its properties and data
        with wave.open(wav_path, 'rb') as wav_file:
            channels = wav_file.getnchannels()

            # Critical check: libopus through pyogg requires 16-bit PCM audio.
            # The toolkit already saves in this format (PCM_S, bits_per_sample=16),
            # but this check ensures correctness.
            if wav_file.getsampwidth() != 2:
                raise PyOggError(f"WAV file must be 16-bit PCM, but '{wav_path}' is not.")

            # 1. Create and fully configure the Opus encoder first.
            encoder = OpusBufferedEncoder()
            # Use the .value attribute of the enum to get the underlying string
            encoder.set_application(application.value)
            encoder.set_sampling_frequency(sampling_rate)
            encoder.set_channels(channels)
            encoder.set_frame_size(frame_size)

            # 2. Create the C-level object.
            encoder.setup_encoder()

            # 3. Set all advanced controls.
            if bitrate is not None:
                encoder.set_ctl(opus.OPUS_SET_BITRATE_REQUEST, bitrate)

            # VBR must be passed as an integer (1 for True, 0 for False)
            encoder.set_ctl(opus.OPUS_SET_VBR_REQUEST, 1 if vbr else 0)

            if complexity is not None:
                if not 0 <= complexity <= 10:
                    raise ValueError("Complexity must be an integer between 0 and 10.")
                encoder.set_ctl(opus.OPUS_SET_COMPLEXITY_REQUEST, complexity)

            # 3. Initialize the OggOpus writer with the fully configured encoder
            writer = OggOpusWriter(encoded_path, encoder)

            # 3. Read the WAV in chunks and write to the encoder
            chunk_size_frames = 4096  # Read in chunks for efficiency
            while True:
                pcm_chunk = wav_file.readframes(chunk_size_frames)
                if not pcm_chunk:
                    break
                # The writer handles buffering and encoding the PCM chunk
                writer.write(memoryview(pcm_chunk))

        # 4. IMPORTANT: Close the writer to finalize the Ogg stream and write metadata
        writer.close()

    except (PyOggError, ValueError) as e:
        print(f"An error occurred during Opus encoding: {e}")
        raise
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        raise

    return encoded_path


def decode_opus(
        opus_encoded_path: str,
        output_folder: str,
        count: str = "",
        decoded_path: str | None = None
) -> str:
    """
    Decodes an OggOpus file back to a WAV file using the vendored pyogg_encoder.

    :param opus_encoded_path: Path to the input Opus file.
    :param output_folder: Directory to save the decoded WAV file.
    :param count: A prefix for the output filename, typically used in bulk generation.
    :param decoded_path: An exact output filename. If None, a name is generated.
    :return: The path to the decoded WAV file.
    :raises PyOggError: If an error occurs during the decoding process.
    """
    # Define the output path for the decoded WAV file
    if decoded_path is None:
        base_name = os.path.basename(opus_encoded_path)
        wav_name = base_name.replace('.opus', '_opus_decoded.wav')
        final_decoded_path = os.path.join(output_folder, f"{count}_{wav_name}")
    else:
        final_decoded_path = os.path.join(output_folder, decoded_path)

    try:
        # 1. Open the input Opus file as a stream
        opus_stream = OpusFileStream(opus_encoded_path)

        # 2. Open the output WAV file for writing
        with wave.open(final_decoded_path, 'wb') as wav_file:
            # Configure WAV file properties from the Opus stream info
            wav_file.setnchannels(opus_stream.channels)
            wav_file.setsampwidth(2)  # Decoded Opus is always 16-bit PCM
            wav_file.setframerate(opus_stream.frequency)

            # 3. Read decoded PCM buffers and write them to the WAV file
            while True:
                buffer = opus_stream.get_buffer()
                if buffer is None:
                    # End of stream
                    break
                wav_file.writeframes(buffer)

    except PyOggError as e:
        print(f"An error occurred during Opus decoding: {e}")
        raise
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        raise

    return final_decoded_path
