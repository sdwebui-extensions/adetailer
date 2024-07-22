from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Union

from huggingface_hub import hf_hub_download
from PIL import Image, ImageDraw
from rich import print
import os

repo_id = "Bingsu/adetailer"
_download_failed = False


@dataclass
class PredictOutput:
    bboxes: list[list[int | float]] = field(default_factory=list)
    masks: list[Image.Image] = field(default_factory=list)
    preview: Optional[Image.Image] = None


def hf_download(file: str):
    global _download_failed

    if _download_failed:
        return "INVALID"

    try:
        path = hf_hub_download(repo_id, file)
    except Exception:
        msg = f"[-] ADetailer: Failed to load model {file!r} from huggingface"
        print(msg)
        path = "INVALID"
        _download_failed = True
    return path


def scan_model_dir(path_: str | Path) -> list[Path]:
    if not path_ or not (path := Path(path_)).is_dir():
        return []
    return [p for p in path.rglob("*") if p.is_file() and p.suffix in (".pt", ".pth")]


def get_models(
    model_dir: str | Path, extra_dir: str | Path = "", huggingface: bool = True
) -> OrderedDict[str, str | None]:
    from modules import shared
    model_paths = [*scan_model_dir(model_dir), *scan_model_dir(extra_dir)]
    if os.path.exists('/stable-diffusion-cache/models/adetailer'):
        for model_name in os.listdir('/stable-diffusion-cache/models/adetailer'):
            if not shared.cmd_opts.just_ui:
                os.system(f'cp /stable-diffusion-cache/models/adetailer/{model_name} {model_dir}/{model_name}')
            model_paths.append(Path(f'{model_dir}/{model_name}'))

    models = OrderedDict()
    if huggingface:
        for model_name in ['face_yolov8n.pt', 'face_yolov8s.pt', 'hand_yolov8n.pt', 'person_yolov8n-seg.pt', 'person_yolov8s-seg.pt']:
            if os.path.exists(os.path.join(model_dir, model_name)):
                continue
            elif os.path.exists(os.path.join('/stable-diffusion-cache/models/adetailer', model_name)):
                if not shared.cmd_opts.just_ui:
                    os.system(f'cp /stable-diffusion-cache/models/adetailer/{model_name} {model_dir}/{model_name}')
                model_paths.append(Path(f'{model_dir}/{model_name}'))
            else:
                models.update({model_name: hf_download(model_name)})
    models.update(
        {
            "mediapipe_face_full": None,
            "mediapipe_face_short": None,
            "mediapipe_face_mesh": None,
            "mediapipe_face_mesh_eyes_only": None,
        }
    )

    invalid_keys = [k for k, v in models.items() if v == "INVALID"]
    for key in invalid_keys:
        models.pop(key)

    for path in model_paths:
        if path.name in models:
            continue
        models[path.name] = str(path)

    return models


def create_mask_from_bbox(
    bboxes: list[list[float]], shape: tuple[int, int]
) -> list[Image.Image]:
    """
    Parameters
    ----------
        bboxes: list[list[float]]
            list of [x1, y1, x2, y2]
            bounding boxes
        shape: tuple[int, int]
            shape of the image (width, height)

    Returns
    -------
        masks: list[Image.Image]
        A list of masks

    """
    masks = []
    for bbox in bboxes:
        mask = Image.new("L", shape, 0)
        mask_draw = ImageDraw.Draw(mask)
        mask_draw.rectangle(bbox, fill=255)
        masks.append(mask)
    return masks


def create_bbox_from_mask(
    masks: list[Image.Image], shape: tuple[int, int]
) -> list[list[int]]:
    """
    Parameters
    ----------
        masks: list[Image.Image]
            A list of masks
        shape: tuple[int, int]
            shape of the image (width, height)

    Returns
    -------
        bboxes: list[list[float]]
        A list of bounding boxes

    """
    bboxes = []
    for mask in masks:
        mask = mask.resize(shape)
        bbox = mask.getbbox()
        if bbox is not None:
            bboxes.append(list(bbox))
    return bboxes
