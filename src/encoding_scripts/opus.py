import logging
import os

from pyogg_encoder import OpusApplication
from pyogg_encoder import PyOggError
from pyogg_encoder import decode_opus as pyogg_decode_opus
from pyogg_encoder import encode_opus as pyogg_encode_opus


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
        pyogg_encode_opus(
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


def opus_to_wav(
        opus_input_path: str,
        wav_output_path: str,
        overwrite: bool = False,
) -> None:
    """
    Converts an OggOpus file to a WAV file.

    This is a simplified wrapper around the core `decode_opus` function.

    :param opus_input_path: Path to the input Opus file.
    :param wav_output_path: The full path where the output WAV file will be saved.
    :param overwrite: If True, will overwrite the output file if it already exists.
    :raises PyOggError: If an error occurs during the decoding process.
    :raises FileNotFoundError: If the input Opus file does not exist.
    """
    # check input file
    if not os.path.exists(opus_input_path):
        raise FileNotFoundError(f"Input file not found: {opus_input_path}")
    if not os.path.isfile(opus_input_path):
        raise IsADirectoryError(f"Input file is not a file: {opus_input_path}")

    # The core `decode_opus` function is designed to take an output folder
    # and a filename, which maps perfectly to our needs.
    output_dir = os.path.dirname(os.path.abspath(wav_output_path))
    output_filename = os.path.basename(wav_output_path)
    os.makedirs(output_dir, exist_ok=True)

    # check output file
    if os.path.exists(wav_output_path):
        if not os.path.isfile(wav_output_path):
            raise IsADirectoryError(f"Output file is not a file: {wav_output_path}")
        elif not overwrite:
            raise FileExistsError(
                f"Output file already exists: '{wav_output_path}'. "
                "Use `overwrite=True` to replace it."
            )
        else:
            os.remove(wav_output_path)

    # convert opus to wav
    try:
        pyogg_decode_opus(
            opus_encoded_path=opus_input_path,
            output_folder=output_dir,
            decoded_path=output_filename
        )
    except PyOggError:
        logging.exception(f"Failed to convert {opus_input_path} to WAV")
        raise


def encode_opus(wav_path: str, tmp_folder: str) -> str:
    """
    transcodes a wav file to opus using 24k bitrate.
    the filename is retained (suffix changes from `*.wav` to `*.opus`),
    and the file is placed into the specified `tmp_folder` dir.

    :param wav_path: path to input wav file
    :param tmp_folder: output folder path
    :return: path to new opus file
    """
    encoded_path = os.path.join(tmp_folder, wav_path.replace('.wav', '.opus'))

    # Use ffmpeg to encode WAV file to Opus
    # command = f"echo y | ffmpeg -i {wav_path} -c:a libopus -b:a {bit_rate} -ar 16000 -nostats -hide_banner -loglevel error {encoded_path}"
    # subprocess.run(command, shell=True, check=True)

    # use libopus (via pyogg) instead
    wav_to_opus(wav_input_path=wav_path,
                opus_output_path=encoded_path,
                bitrate=24000,
                overwrite=True)
    return encoded_path


def decode_opus(opus_encoded_path: str,
                output_folder: str,
                count: str = "",
                decoded_path: str | None = None,
                ) -> str:
    """
    transcodes an opus file to wav.
    the filename is retained (suffix changes from `*.opus` to `{count}_*_opus_decoded.wav`),
    unless a new filename is specified via `decoded_path`.
    note: count defaults to an empty string, so the default output name is `_*_opus_decoded.wav`.
    the file is placed into the specified `output_folder` dir.

    :param opus_encoded_path: path to input opus file
    :param output_folder: output folder path
    :param count: prefix for the output file name
    :param decoded_path: if provided, is used instead of the default generated output file name
    :return: path to new wav file
    """

    if decoded_path is None:
        decoded_path = count + "_" + os.path.split(opus_encoded_path)[-1].replace('.opus', '_opus_decoded.wav')
    decoded_path = os.path.join(output_folder, decoded_path)

    # Use ffmpeg to decode the Opus file back to WAV
    # encoding = "pcm_s16le"
    # sampling_rate = 16000
    # control encoding "-c:a pcm_s16le"; sampling_rate with "-ar 48000"
    # echo y to overwrite by default
    # command = f"echo y | ffmpeg -i {opus_encoded_path} -c:a {encoding} -ar {sampling_rate} -nostats -hide_banner -loglevel error {decoded_path}"
    # subprocess.run(command, shell=True, check=True)

    # use libopus (via pyogg)
    # note that `opus_to_wav` output is always 16-bit pcm
    opus_to_wav(opus_input_path=opus_encoded_path,
                wav_output_path=decoded_path,
                overwrite=True)
    return decoded_path


def main():
    input_wav_path = 'input.wav'

    # Encode the WAV file to Opus
    encoded_path = encode_opus(input_wav_path, '.')
    print(f"Encoded to: {encoded_path}")

    # Decode the Opus encoded audio back to WAV
    decoded_path = decode_opus(encoded_path, '.')
    print(f"Decoded to: {decoded_path}")


if __name__ == "__main__":
    main()
