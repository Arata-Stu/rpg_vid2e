import os
from pathlib import Path
from typing import Union

from fractions import Fraction
from PIL import Image
import numpy as np

# skvideo (used for ffprobe/vreader) still references deprecated numpy aliases
# on some releases. Restore aliases for numpy>=1.24 compatibility.
_NUMPY_ALIAS_COMPAT = {
    "float": float,
    "int": int,
    "bool": bool,
    "complex": complex,
    "object": object,
    "str": str,
}
for _alias, _type in _NUMPY_ALIAS_COMPAT.items():
    if _alias not in np.__dict__:
        setattr(np, _alias, _type)  # type: ignore[attr-defined]

import skvideo.io

from .const import mean, std, img_formats


class Sequence:
    def __init__(self):
        pass

    def __iter__(self):
        return self

    def __next__(self):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError


def _pil_loader(path: str):
    with open(path, 'rb') as f:
        img = Image.open(f)
        img = img.convert('RGB')

        w_orig, h_orig = img.size
        w, h = w_orig//32*32, h_orig//32*32

        left = (w_orig - w)//2
        upper = (h_orig - h)//2
        right = left + w
        lower = upper + h
        img = img.crop((left, upper, right, lower))
        return np.array(img).astype("float32") / 255


class ImageSequence(Sequence):
    def __init__(self, imgs_dirpath: str, fps: float):
        super().__init__()
        self.fps = fps

        assert os.path.isdir(imgs_dirpath)
        self.imgs_dirpath = imgs_dirpath

        self.file_names = [f for f in os.listdir(imgs_dirpath) if self._is_img_file(f)]
        assert self.file_names
        self.file_names.sort()

    @classmethod
    def _is_img_file(cls, path: str):
        return Path(path).suffix.lower() in img_formats

    def __next__(self):
        for idx in range(0, len(self.file_names) - 1):
            file_paths = self._get_path_from_name([self.file_names[idx], self.file_names[idx + 1]])
            imgs = [_pil_loader(f) for f in file_paths]
            times_sec = [idx/self.fps, (idx + 1)/self.fps]
            yield imgs, times_sec

    def __len__(self):
        return len(self.file_names) - 1

    def _get_path_from_name(self, file_names: Union[list, str]) -> Union[list, str]:
        if isinstance(file_names, list):
            return [os.path.join(self.imgs_dirpath, f) for f in file_names]
        return os.path.join(self.imgs_dirpath, file_names)


class ManifestSequence(Sequence):
    """
    Sequence backed by a manifest text file.

    Format per non-empty line:
      <timestamp_seconds> <absolute_or_relative_image_path>
    """

    def __init__(self, manifest_filepath: str):
        super().__init__()
        assert os.path.isfile(manifest_filepath), manifest_filepath
        self.manifest_filepath = os.path.abspath(manifest_filepath)

        rows = []
        with open(self.manifest_filepath, "r") as f:
            for line_num, raw_line in enumerate(f, 1):
                line = raw_line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split(maxsplit=1)
                if len(parts) != 2:
                    raise ValueError(
                        f"Invalid manifest line {line_num} in {self.manifest_filepath}: "
                        f"expected '<timestamp> <path>'"
                    )
                t_str, img_path = parts
                timestamp = float(t_str)
                if not os.path.isabs(img_path):
                    img_path = os.path.join(os.path.dirname(self.manifest_filepath), img_path)
                rows.append((timestamp, os.path.abspath(img_path)))

        if len(rows) < 2:
            raise ValueError(
                f"Expected at least 2 frames in manifest {self.manifest_filepath}, got {len(rows)}"
            )

        rows.sort(key=lambda x: x[0])
        t0 = rows[0][0]
        self.timestamps_sec = [t - t0 for t, _ in rows]
        self.image_paths = [p for _, p in rows]

        for image_path in self.image_paths:
            if not os.path.isfile(image_path):
                raise FileNotFoundError(f"Image referenced in manifest does not exist: {image_path}")

    def __next__(self):
        for idx in range(0, len(self.image_paths) - 1):
            imgs = [_pil_loader(self.image_paths[idx]), _pil_loader(self.image_paths[idx + 1])]
            times_sec = [self.timestamps_sec[idx], self.timestamps_sec[idx + 1]]
            yield imgs, times_sec

    def __len__(self):
        return len(self.image_paths) - 1


class VideoSequence(Sequence):
    def __init__(self, video_filepath: str, fps: float=None):
        super().__init__()
        metadata = skvideo.io.ffprobe(os.path.abspath(video_filepath))
        self.fps = fps
        if self.fps is None:
            self.fps = float(Fraction(metadata['video']['@avg_frame_rate']))
            assert self.fps > 0, 'Could not retrieve fps from video metadata. fps: {}'.format(self.fps)
            print('Using video metadata: Got fps of {} frames/sec'.format(self.fps))

        # Length is number of frames - 1 (because we return pairs).
        self.len = int(metadata['video']['@nb_frames']) - 1
        self.videogen = skvideo.io.vreader(os.path.abspath(video_filepath))
        self.last_frame = None

    def __next__(self):
        for idx, frame in enumerate(self.videogen):
            h_orig, w_orig, _ = frame.shape
            w, h = w_orig//32*32, h_orig//32*32

            left = (w_orig - w)//2
            upper = (h_orig - h)//2
            right = left + w
            lower = upper + h
            frame = frame[upper:lower, left:right].astype("float32") / 255
            assert frame.shape[:2] == (h, w)

            if self.last_frame is None:
                self.last_frame = frame
                continue

            last_frame_copy = self.last_frame.copy()
            self.last_frame = frame
            imgs = [last_frame_copy, frame]
            times_sec = [(idx - 1)/self.fps, idx/self.fps]
            yield imgs, times_sec

    def __len__(self):
        return self.len
