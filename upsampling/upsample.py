import argparse
import os
# Must be set before importing TensorFlow.
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'


def get_flags():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True, help='Path to input directory. See README.md for expected structure of the directory.')
    parser.add_argument("--output_dir", required=True, help='Path to non-existing output directory. This script will generate the directory.')
    parser.add_argument(
        "--device",
        default="auto",
        help="Device selection for TensorFlow/FILM: auto (default), cpu, or CUDA index (e.g. 0).",
    )
    parser.add_argument(
        "--tf_log_level",
        default="2",
        choices=["0", "1", "2", "3"],
        help=(
            "TensorFlow C++ log level: 0=all, 1=hide INFO, 2=hide INFO/WARN "
            "(default), 3=hide INFO/WARN/ERROR."
        ),
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Silence script-side informational prints (GPU detection message).",
    )
    args = parser.parse_args()
    return args


def _configure_device(device: str):
    if device == "auto":
        return
    if device == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        return
    # Support legacy values like cuda:0 from older scripts.
    if device.startswith("cuda:"):
        device = device.split(":", 1)[1]
    os.environ["CUDA_VISIBLE_DEVICES"] = device


def _configure_tf_logging(tf_log_level: str):
    # Must be set before importing tensorflow.
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = tf_log_level


def main():
    flags = get_flags()
    _configure_device(flags.device)
    _configure_tf_logging(flags.tf_log_level)

    # Import after setting CUDA_VISIBLE_DEVICES.
    import tensorflow as tf
    from utils import Upsampler

    gpus = tf.config.list_physical_devices("GPU")
    if not flags.quiet and flags.device != "cpu" and not gpus:
        print(
            "WARNING: TensorFlow did not detect a GPU and will run on CPU. "
            "If you want GPU upsampling, verify TensorFlow CUDA/cuDNN runtime alignment."
        )
    elif not flags.quiet and gpus:
        print(f"TensorFlow detected {len(gpus)} GPU(s): {[gpu.name for gpu in gpus]}")

    upsampler = Upsampler(input_dir=flags.input_dir, output_dir=flags.output_dir)
    upsampler.upsample()


if __name__ == '__main__':
    main()
