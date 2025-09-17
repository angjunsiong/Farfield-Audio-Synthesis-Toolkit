# Far-field Audio Synthesis Toolkit (FAST)

## Installation

* Python 3.10
  * `conda create -n fast python=3.10`
  * `conda activate fast`
* Install ffmpeg: `conda install pytorch::ffmpeg`
* `pip install -r requirements.txt`
* Install `rubberband-cli` from https://breakfastquay.com/rubberband/
  * Make sure to add it to your PATH and check that you can call `rubberband` in command prompt
  * Or put it in (something like) `%USERPROFILE%\anaconda3\envs\fast\Library\bin`

## Usage

* Start with [augment_playbook_RIR.ipynb](./augment_playbook_RIR.ipynb)