import os

from pyogg_encoder import PyOggError
from pyogg_encoder import decode_opus


def opus_to_wav(
        opus_input_path: str,
        wav_output_path: str
) -> None:
    """
    Converts an OggOpus file to a WAV file.

    This is a simplified wrapper around the core `decode_opus` function.

    :param opus_input_path: Path to the input Opus file.
    :param wav_output_path: The full path where the output WAV file will be saved.
    :raises PyOggError: If an error occurs during the decoding process.
    :raises FileNotFoundError: If the input Opus file does not exist.
    """
    # The core `decode_opus` function is designed to take an output folder
    # and a filename, which maps perfectly to our needs.
    output_dir = os.path.dirname(os.path.abspath(wav_output_path))
    output_filename = os.path.basename(wav_output_path)
    os.makedirs(output_dir, exist_ok=True)

    if not os.path.exists(opus_input_path):
        raise FileNotFoundError(f"Input file not found: {opus_input_path}")

    try:
        decode_opus(
            opus_encoded_path=opus_input_path,
            output_folder=output_dir,
            decoded_path=output_filename
        )
    except PyOggError as e:
        print(f"Failed to convert {opus_input_path} to WAV: {e}")
        raise


if __name__ == '__main__':
    opus_to_wav("sample1.opus", "sample1.wav")
