# Setup Guide (Split Environments)

This project is most stable when TensorFlow and PyTorch are installed in **separate virtual environments**.

Use these explicit names (no leading dot):
- `venv_vid2e_torch`: PyTorch, `esim_torch`, event generation, web app runtime
- `venv_vid2e_tf`: TensorFlow/FILM upsampling

## 0. Host prerequisites

- Linux + NVIDIA driver (`nvidia-smi` works)
- CUDA Toolkit installed on host (`nvcc` available) for building `esim_torch`
- Python 3.10 or 3.11

## 1. Clone repository

```bash
git clone git@github.com:uzh-rpg/rpg_vid2e.git --recursive
cd rpg_vid2e
```

## 2. Create both environments

```bash
python3 -m venv venv_vid2e_torch
python3 -m venv venv_vid2e_tf
```

## 3. Setup `venv_vid2e_torch` (PyTorch / esim)

```bash
source venv_vid2e_torch/bin/activate
pip install -U pip setuptools wheel
pip install -r requirements.txt
pip install -r requirements-webapp.txt

# Example: CUDA 12.6 wheels
pip uninstall -y torch torchvision torchaudio
pip install --index-url https://download.pytorch.org/whl/cu126 torch torchvision torchaudio
pip install ninja
```

Align toolkit for extension build (example for `+cu126`):

```bash
export CUDA_HOME=/usr/local/cuda-12.6
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}
hash -r
```

Check alignment:

```bash
which nvcc
nvcc -V
python3 - <<'PY'
import torch
from torch.utils.cpp_extension import CUDA_HOME
print('torch:', torch.__version__)
print('torch.version.cuda:', torch.version.cuda)
print('cpp_extension.CUDA_HOME:', CUDA_HOME)
PY
```

Build local packages:

```bash
pip install -e ./esim_torch --no-build-isolation
pip install -e ./esim_py --no-build-isolation
```

Deactivate when done:

```bash
deactivate
```

## 4. Setup `venv_vid2e_tf` (TensorFlow / upsampling)

```bash
source venv_vid2e_tf/bin/activate
pip install -U pip setuptools wheel
pip install -r requirements-upsampling.txt
pip uninstall -y tensorflow tensorflow-cpu tensorflow-intel
pip install "tensorflow[and-cuda]==2.17.1"
```

Check GPU visibility:

```bash
python3 - <<'PY'
import tensorflow as tf
print('tf:', tf.__version__)
print('gpus:', tf.config.list_physical_devices('GPU'))
PY
```

Deactivate when done:

```bash
deactivate
```

## 5. Run pipeline with split envs

### 5.1 Upsampling (TensorFlow env)

```bash
source venv_vid2e_tf/bin/activate
python3 upsampling/upsample.py \
  --input_dir ./example/original \
  --output_dir ./example/upsampled \
  --device=0 \
  --tf_log_level=3 \
  --quiet
deactivate
```

### 5.2 Event generation (PyTorch env)

```bash
source venv_vid2e_torch/bin/activate
python3 esim_torch/scripts/generate_events.py \
  --input_dir=example/upsampled \
  --output_dir=example/events \
  --contrast_threshold_neg=0.2 \
  --contrast_threshold_pos=0.2 \
  --refractory_period_ns=0 \
  --device=cuda
deactivate
```

## 6. Web app with split envs

Run Streamlit in PyTorch env, and tell it to invoke upsampling with TensorFlow env python:

```bash
source venv_vid2e_torch/bin/activate
export UPSAMPLING_PYTHON="$(pwd)/venv_vid2e_tf/bin/python"
cd web_app
streamlit run web_app.py
```

## 7. Troubleshooting

### A. `CUDA_MISMATCH_MESSAGE` while building `esim_torch`

Cause: `nvcc` CUDA version and `torch.version.cuda` differ.

Fix:
1. Reinstall matching PyTorch wheel (`cu126`, etc.)
2. Export matching `CUDA_HOME`
3. Re-run `pip install -e ./esim_torch --no-build-isolation`

### B. TensorFlow `gpus: []`

Cause: TensorFlow GPU runtime not resolved.

Fix:
1. Reinstall `tensorflow[and-cuda]==2.17.1` in `venv_vid2e_tf`
2. Verify driver with `nvidia-smi`
3. Re-run TensorFlow GPU check

CPU fallback remains available:

```bash
source venv_vid2e_tf/bin/activate
python3 upsampling/upsample.py \
  --input_dir ./example/original \
  --output_dir ./example/upsampled_cpu \
  --device=cpu \
  --tf_log_level=3 \
  --quiet
```
