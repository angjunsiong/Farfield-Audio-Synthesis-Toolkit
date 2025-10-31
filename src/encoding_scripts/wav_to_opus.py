import logging
import os

from pyogg_encoder import OpusApplication
from pyogg_encoder import PyOggError
from pyogg_encoder import encode_opus


def wav_to_opus(
        wav_input_path: str,
        opus_output_path: str,
        application: OpusApplication = OpusApplication.AUDIO,
        bitrate: int | None = 64000,
        vbr: bool = True,
        complexity: int | None = 10,
        overwrite: bool = False,
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
    :param overwrite: If True, will overwrite the output file if it already exists.
    :raises PyOggError: If an error occurs during the encoding process.
    :raises FileNotFoundError: If the input WAV file does not exist.
    :raises FileExistsError: If the output file exists and overwrite is False.
    """
    if not os.path.exists(wav_input_path):
        raise FileNotFoundError(f"Input file not found: {wav_input_path}")
    if not os.path.isfile(wav_input_path):
        raise IsADirectoryError(f"Input file is not a file: {wav_input_path}")

    # The core `encode_opus` function needs a directory to write its output,
    # which it names based on the input file. We use the desired output
    # directory for this temporary step.
    output_dir = os.path.dirname(os.path.abspath(opus_output_path))
    os.makedirs(output_dir, exist_ok=True)

    if os.path.exists(opus_output_path):
        if not os.path.isfile(opus_output_path):
            raise IsADirectoryError(f"Output file is not a file: {opus_output_path}")
        elif not overwrite:
            raise FileExistsError(
                f"Output file already exists: '{opus_output_path}'. "
                "Use `overwrite=True` to replace it."
            )
        else:
            os.remove(opus_output_path)

    # convert wav to opus
    try:
        encode_opus(
            wav_path=wav_input_path,
            output_opus_path=opus_output_path,
            application=application,
            bitrate=bitrate,
            vbr=vbr,
            complexity=complexity
        )
    except (PyOggError, ValueError):
        logging.exception(f"Failed to convert {wav_input_path} to Opus")
        raise


if __name__ == '__main__':
    wav_to_opus("sample1.wav", "sample1_converted.opus", overwrite=True)
