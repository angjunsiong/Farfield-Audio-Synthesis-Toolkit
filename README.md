# Far-field Audio Synthesis Toolkit (FAST)

## Installation

* Python 3.10
    * `conda create -n fast python=3.10`
    * `conda activate fast`
* Install ffmpeg: `conda install pytorch::ffmpeg`
    * Or `conda install ./install/pytorch-ffmpeg-windows-4.3-ha925a31_0.tar.bz2` if you're behind a stupid proxy
* `pip install -r requirements.txt`
* Install `rubberband-cli` from https://breakfastquay.com/rubberband/
    * Also at `./install/rubberband-4.0.0-gpl-executable-windows.zip`
    * Make sure to add it to your PATH and check that you can call `rubberband` in command prompt
    * Or put it in (something like) `%USERPROFILE%\anaconda3\envs\fast\Library\bin`

## Usage

* Start with [augment_playbook_RIR.ipynb](./augment_playbook_RIR.ipynb)