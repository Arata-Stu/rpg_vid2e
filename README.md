# Video to Events: Recycling Video Datasets for Event Cameras

<p align="center">
  <a href="https://youtu.be/uX6XknBGg0w">
    <img src="http://rpg.ifi.uzh.ch/data/VID2E/thumb.png" alt="Video to Events" width="600"/>
  </a>
</p>

This repository contains code that implements 
video to events conversion as described in Gehrig et al. CVPR'20 and the used dataset. The paper can be found [here](http://rpg.ifi.uzh.ch/docs/CVPR20_Gehrig.pdf)

If you use this code in an academic context, please cite the following work:

[Daniel Gehrig](https://danielgehrig18.github.io/), [Mathias Gehrig](https://magehrig.github.io/), [Javier Hidalgo-Carrió](https://jhidalgocarrio.github.io/), [Davide Scaramuzza](http://rpg.ifi.uzh.ch/people_scaramuzza.html), "Video to Events: Recycling Video Datasets for Event Cameras", The Conference on Computer Vision and Pattern Recognition (CVPR), 2020

```bibtex
@InProceedings{Gehrig_2020_CVPR,
  author = {Daniel Gehrig and Mathias Gehrig and Javier Hidalgo-Carri\'o and Davide Scaramuzza},
  title = {Video to Events: Recycling Video Datasets for Event Cameras},
  booktitle = {{IEEE} Conf. Comput. Vis. Pattern Recog. (CVPR)},
  month = {June},
  year = {2020}
}
```
## News
* We now support frame interpolation done by [FILM](https://github.com/google-research/frame-interpolation).
* We release a web app and interactive demo which generates events and converts your webcam to events. Try it out [here](web_app/README.md).
* We now also release new python bindings for esim with GPU support.
Details are [here](esim_torch/README.md)

## Web App and Interactive Demo
Try out our the interactive demo and webcam support [here](web_app/README.md). 

## Dataset
The synthetic N-Caltech101 dataset, as well as video sequences used for event conversion can be found [here](http://rpg.ifi.uzh.ch/data/VID2E/ncaltech_syn_images.zip). For each sample of each class it contains events in the form `class/image_%04d.npz` and images in the form `class/image_%05d/images/image_%05d.png`, as well as the corresponding timestamps of the images in `class/image_%04d/timestamps.txt`.

## Installation
Clone the repo *recursively with submodules*

```bash
git clone git@github.com:uzh-rpg/rpg_vid2e.git --recursive
cd rpg_vid2e
```

For the recommended split-environment setup (`vid2e_env` + `film_env`), see [SETUP.md](SETUP.md).

### Torch Environment (modern GPU)

Create an environment with Python 3.10+ and install the base dependencies:

```bash
python3 -m venv vid2e_env
source vid2e_env/bin/activate
pip install -U pip setuptools wheel
pip install -r requirements.txt
```

Install a CUDA-enabled PyTorch build that matches your local driver/runtime (see [PyTorch install selector](https://pytorch.org/get-started/locally/)).

```bash
pip install torch torchvision torchaudio
```

Before building `esim_torch`, verify that PyTorch CUDA and your local CUDA toolkit are aligned:

```bash
echo "CUDA_HOME=$CUDA_HOME"
which nvcc
python3 - <<'PY'
import torch
from torch.utils.cpp_extension import CUDA_HOME
print("torch:", torch.__version__)
print("torch.version.cuda:", torch.version.cuda)
print("cpp_extension.CUDA_HOME:", CUDA_HOME)
PY
nvcc -V
```

If your shell does not pick the intended CUDA toolkit, add persistent CUDA path exports in `~/.bashrc` as described in [SETUP.md](SETUP.md) (`0.1 Persist CUDA path in ~/.bashrc`).

If you hit `RuntimeError: The detected CUDA version mismatches the version that was used to compile PyTorch`, reinstall a matching PyTorch wheel and retry. Example for CUDA 12.6:

```bash
pip uninstall -y torch torchvision torchaudio
pip install --index-url https://download.pytorch.org/whl/cu126 torch torchvision torchaudio
pip install ninja
pip install -e ./esim_torch --no-build-isolation
```

Build/install the GPU bindings:

```bash
pip install -e ./esim_torch --no-build-isolation
```

Optionally build/install the CPU pybind bindings:

```bash
pip install -e ./esim_py --no-build-isolation
```

### Optional: FILM upsampling dependencies

Download the [FILM](https://github.com/google-research/frame-interpolation) checkpoint and place it at `pretrained_models/film_net/Style/saved_model`:

```bash
wget https://rpg.ifi.uzh.ch/data/VID2E/pretrained_models.zip -O /tmp/temp.zip
unzip /tmp/temp.zip -d .
rm -rf /tmp/temp.zip
```

Install optional upsampling dependencies:

```bash
python3 -m venv film_env
source film_env/bin/activate
pip install -r requirements-upsampling.txt
pip uninstall -y tensorflow tensorflow-cpu tensorflow-intel
pip install "tensorflow[and-cuda]==2.17.1"
```

For historical reproducibility with the original pinned dependency set, use `requirements-legacy.txt`.

## Adaptive Upsampling
*This package provides code for adaptive upsampling with frame interpolation based on [Super-SloMo](https://people.cs.umass.edu/~hzjiang/projects/superslomo/)*

Consult the [README](upsampling/README.md) for detailed instructions and examples.

## esim\_py
*This package exposes python bindings for [ESIM](http://rpg.ifi.uzh.ch/docs/CORL18_Rebecq.pdf) which can be used within a training loop.*

For detailed instructions and example consult the [README](esim_py/README.md)

## esim\_torch
*This package exposes python bindings for [ESIM](http://rpg.ifi.uzh.ch/docs/CORL18_Rebecq.pdf) with GPU support.*

For detailed instructions and example consult the [README](esim_torch/README.md)

## Example
To run an example, first upsample the example videos 

```bash
source film_env/bin/activate
device=auto
# device=cpu
# device=0
python3 upsampling/upsample.py --input_dir=example/original --output_dir=example/upsampled --device=$device --tf_log_level=2
deactivate
```
This will generate upsampled frames in the `example/upsampled` folder. To generate events, use
```bash
source vid2e_env/bin/activate
python3 esim_torch/scripts/generate_events.py --input_dir=example/upsampled \
                                             --output_dir=example/events \
                                             --contrast_threshold_neg=0.2 \
                                             --contrast_threshold_pos=0.2 \
                                             --refractory_period_ns=0 \
                                             --device=cuda
deactivate
```
