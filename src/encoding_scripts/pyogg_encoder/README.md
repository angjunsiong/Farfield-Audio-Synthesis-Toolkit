# Vendored PyOgg

* code copied from https://github.com/TeamPyOgg/PyOgg
* vendored instead of installed (via requirements) because they haven't published the newer version to pypi,
  and we need features from the new code to convert `.wav` back to `.opus`
* minor changes from gemini to support advanced quality params (bitrate, vbr, complexity)