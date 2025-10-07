# Vendored `PyOgg`

* code copied from https://github.com/TeamPyOgg/PyOgg
  at [commit `4118fc4`](https://github.com/TeamPyOgg/PyOgg/commit/4118fc40067eb475468726c6bccf1242abfc24fc)
* vendored instead of installed (via [requirements.txt](../../../requirements.txt)) because:
    1. they haven't published the newer version to pypi
    2. we need features from the new code to convert `.wav` back to `.opus`
    3. minor changes (via gemini) were needed to support advanced quality params (bitrate, vbr, complexity)