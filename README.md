# Far-field Audio Synthesis Toolkit (FAST)

## Installation

* Python 3.10
    * `conda create -n fast python=3.10`
    * `conda activate fast`
* `pip install -r requirements.txt`
* Install `rubberband-cli` from [`./install`](./install)
    * Or get it from https://breakfastquay.com/rubberband
    * Make sure to add it to your PATH and check that you can call `rubberband` in command prompt
    * If you're using conda, all the binaries should go somewhere like `%USERPROFILE%\anaconda3\envs\fast\Library\bin`
    * For Linux x86_64, it's not ideal but the current build dynamically links to global library dependency. If you're having troubles, just rebuild from the source repo and put the binary in whatever (possibly conda) path you use. Below is the output of ldd:
    ```
        linux-vdso.so.1 (0x000070f05e195000)
        libsndfile.so.1 => /lib/x86_64-linux-gnu/libsndfile.so.1 (0x000070f05e082000)
        libstdc++.so.6 => /lib/x86_64-linux-gnu/libstdc++.so.6 (0x000070f05de00000)
        libm.so.6 => /lib/x86_64-linux-gnu/libm.so.6 (0x000070f05dd17000)
        libgcc_s.so.1 => /lib/x86_64-linux-gnu/libgcc_s.so.1 (0x000070f05dce9000)
        libc.so.6 => /lib/x86_64-linux-gnu/libc.so.6 (0x000070f05da00000)
        libFLAC.so.12 => /lib/x86_64-linux-gnu/libFLAC.so.12 (0x000070f05dc85000)
        libvorbis.so.0 => /lib/x86_64-linux-gnu/libvorbis.so.0 (0x000070f05dc57000)
        libvorbisenc.so.2 => /lib/x86_64-linux-gnu/libvorbisenc.so.2 (0x000070f05d955000)
        libopus.so.0 => /lib/x86_64-linux-gnu/libopus.so.0 (0x000070f05d8f6000)
        libogg.so.0 => /lib/x86_64-linux-gnu/libogg.so.0 (0x000070f05dc4d000)
        libmpg123.so.0 => /lib/x86_64-linux-gnu/libmpg123.so.0 (0x000070f05d89a000)
        libmp3lame.so.0 => /lib/x86_64-linux-gnu/libmp3lame.so.0 (0x000070f05d824000)
        /lib64/ld-linux-x86-64.so.2 (0x000070f05e197000)
    ```
    

## Usage

* Start with [augment_playbook_RIR.ipynb](./augment_playbook_RIR.ipynb)