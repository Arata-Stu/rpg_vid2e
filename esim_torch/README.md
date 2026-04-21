# esim\_torch

This package exposes python bindings for ESIM with GPU support.

## Build / Install

Install a CUDA-enabled PyTorch build first (matching your local driver/CUDA runtime), then build this extension:

```bash
pip install -e ./esim_torch --no-build-isolation
```

`--no-build-isolation` is recommended so the build uses the same PyTorch install as your environment.

## CUDA mismatch troubleshooting

If build fails with a CUDA mismatch error, check your runtime/toolkit alignment:

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

If versions do not match, reinstall a matching PyTorch wheel and rebuild. Example for CUDA 12.6:

```bash
pip uninstall -y torch torchvision torchaudio
pip install --index-url https://download.pytorch.org/whl/cu126 torch torchvision torchaudio
pip install ninja
pip install -e ./esim_torch --no-build-isolation
```

## Quick test

Test your installation with

```bash
python3 esim_torch/test/test.py
```

which should create a plot.

The currently supported functions are listed in the example below:
```python
import esim_torch

# constructor
esim = esim_torch.ESIM(
    contrast_threshold_neg,  # contrast threshold for negative events
    contrast_threshold_pos,  # contrast threshold for positive events
    refractory_period_ns     # refractory period in nanoseconds
)

# event generation
events = esim.forward(
    log_images,        # torch tensor with type float32, shape T x H x W
    timestamps_ns  # torch tensor with type int64,   shape T 
)

# Reset the internal state of the simulator
esim.reset()

# events can also be generated in a for loop 
# to keep memory requirements low
for log_image, timestamp_ns in zip(log_images, timestamps_ns):
    sub_events = esim.forward(log_image, timestamp_ns)

    # for the first image, no events are generated, so this needs to be skipped
    if sub_events is None:
        continue

    # do something with the events
    some_function(sub_events)

```
