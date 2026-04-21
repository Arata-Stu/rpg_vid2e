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


def main():
    flags = get_flags()
    _configure_device(flags.device)

    # Import after setting CUDA_VISIBLE_DEVICES.
    from utils import Upsampler

    upsampler = Upsampler(input_dir=flags.input_dir, output_dir=flags.output_dir)
    upsampler.upsample()


if __name__ == '__main__':
    main()
