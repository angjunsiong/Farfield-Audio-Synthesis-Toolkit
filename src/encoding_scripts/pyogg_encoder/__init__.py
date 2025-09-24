# This file is part of a local module created to vendor encoding functionality
# from the newer TeamPyOgg/PyOgg repository (github.com/TeamPyOgg/PyOgg).
# It allows WAV-to-Opus conversion without modifying the system's PyOgg installation.

# Expose the necessary classes for easy importing
from .ogg_opus_writer import OggOpusWriter
from .opus_buffered_encoder import OpusBufferedEncoder
from .opus_file_stream import OpusFileStream
from .pyogg_error import PyOggError
