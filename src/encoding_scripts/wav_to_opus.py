import os

from pyogg_encoder import OpusApplication
from pyogg_encoder import PyOggError
from pyogg_encoder import encode_opus


def wav_to_opus(
        wav_input_path: str,
        opus_output_path: str,
        application: OpusApplication = OpusApplication.AUDIO,
        bitrate: int | None = 128000,
        vbr: bool = True,
        complexity: int | None = 10
) -> None:
    """
    Converts a WAV file to an OggOpus file with specified quality settings.

    This is a simplified wrapper around the core `encode_opus` function.

    :param wav_input_path: Path to the input 16-bit WAV file.
    :param opus_output_path: The full path where the output Opus file will be saved.
    :param application: The Opus application mode (AUDIO, VOIP, etc.).
    :param bitrate: Target bitrate in bits per second.
    :param vbr: Whether to use Variable Bitrate.
    :param complexity: Encoder complexity (0-10).
    :raises PyOggError: If an error occurs during the encoding process.
    :raises FileNotFoundError: If the input WAV file does not exist.
    """
    # The core `encode_opus` function needs a directory to write its output,
    # which it names based on the input file. We use the desired output
    # directory for this temporary step.
    output_dir = os.path.dirname(os.path.abspath(opus_output_path))
    os.makedirs(output_dir, exist_ok=True)

    if not os.path.exists(wav_input_path):
        raise FileNotFoundError(f"Input file not found: {wav_input_path}")

    try:
        # Call the core encoder. It will place a file like 'input_name.opus'
        # into the output_dir.
        temp_encoded_path = encode_opus(
            wav_path=wav_input_path,
            tmp_folder=output_dir,
            application=application,
            bitrate=bitrate,
            vbr=vbr,
            complexity=complexity
        )

        # Rename the generated file to the exact output path requested by the user.
        # This handles the case where the auto-generated name is not the desired name.
        if temp_encoded_path != opus_output_path:
            os.rename(temp_encoded_path, opus_output_path)

    except (PyOggError, ValueError) as e:
        print(f"Failed to convert {wav_input_path} to Opus: {e}")
        raise


if __name__ == '__main__':
    wav_to_opus("sample1.wav", "sample1_converted.opus")
