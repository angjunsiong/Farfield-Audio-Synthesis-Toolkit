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
        output_opus_path: str,
        application: OpusApplication = OpusApplication.AUDIO,
        frame_size: int = 20,
        bitrate: int | None = None,
        vbr: bool = True,
        complexity: int | None = None,
        force_sampling_rate: int | None = None,
) -> str:
    """
    Encodes a WAV file to an OggOpus file using the vendored pyogg_encoder.

    This function provides direct access to libopus settings for higher quality
    encoding, allowing for explicit control over the SILK (speech) and CELT
    (music) layers of the Opus codec.

    See also: https://opus-codec.org/docs/opus_api-1.5/group__opus__encoderctls.html

    :param wav_path: Path to the input WAV file. Must be 16-bit PCM.
    :param output_opus_path: The full, final path for the output Opus file.
    :param application: The Opus application mode to use for encoding.
    :param frame_size: Opus frame size in milliseconds (one of {25, 50, 100, 200, 400, 600}).
    :param bitrate: Target bitrate in bits per second (between 500 and 512000). If None, the codec's default is used.
    :param vbr: If True, enables Variable Bitrate for better quality.
    :param complexity: Encoder complexity (0-10). 10 is highest quality but slowest.
    :param force_sampling_rate: Override the sampling rate of the audio in Hz (e.g., 16000).
    :return: The path to the encoded Opus file (the same as output_opus_path).
    """
    # --- SIMPLIFIED LOGIC ---
    # The function now writes directly to the final destination.
    encoded_path = output_opus_path

    # Ensure the destination directory exists.
    output_dir = os.path.dirname(os.path.abspath(encoded_path))
    os.makedirs(output_dir, exist_ok=True)

    try:
        # Open the input WAV file to read its properties and data
        with wave.open(wav_path, 'rb') as wav_file:
            channels = wav_file.getnchannels()

            # Critical check: libopus through pyogg requires 16-bit PCM audio.
            # The toolkit already saves in this format (PCM_S, bits_per_sample=16),
            # but this check ensures correctness.
            if wav_file.getsampwidth() != 2:
                raise PyOggError(f"WAV file must be 16-bit PCM, but '{wav_path}' is not.")

            # Use the provided sampling_rate if specified, otherwise use the file's rate.
            file_sampling_rate = wav_file.getframerate()
            encoder_sampling_rate = force_sampling_rate if force_sampling_rate is not None else file_sampling_rate

            # 1. Configure the Opus encoder
            encoder = OpusBufferedEncoder()
            encoder.set_application(application.value)
            encoder.set_sampling_frequency(encoder_sampling_rate)
            encoder.set_channels(channels)
            encoder.set_frame_size(frame_size)

            # 2. Create and configure the C-level encoder in a single, safe step
            encoder.setup_encoder(
                bitrate=bitrate,
                vbr=vbr,
                complexity=complexity
            )

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

    except (PyOggError, ValueError) as e:
        print(f"An error occurred during Opus encoding: {e}")
        raise
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        raise
    finally:
        # 4. IMPORTANT: This block ALWAYS runs, ensuring the file is
        # finalized and closed even if an error occurred above.
        try:
            writer.close()
        except Exception as e:
            print(f"Warning: Failed to close OggOpusWriter: {e}")

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
