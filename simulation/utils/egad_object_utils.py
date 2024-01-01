import json
from pathlib import Path

import numpy as np
import sapien.core as sapien

import pdb
#============== add orientation===================
EGAD_LIST = {
    "front": [
        'A2', 'A3', 'C5', 'C1', 'C0', 'E1', 'A6', 'C4', 'E5', 'E4', 'B1', 'C6',
        'E3', 'B4', 'D1', 'D5', 'C2', 'D2', 'F2', 'A4', 'D3', 'D6', 'A1', 'C3',
        'F5', 'B5', 'A0', 'F1', 'B2', 'F4', 'B0', 'A5', 'B3', 'B6', 'F3', 'E6',
        'D4', 'D0'
    ],
    "down": ["C0", "D1", "D2"]
}
EGAD_ORIENTATION = {
    "A0": (0.707, -0.707, 0, 0),
    "A1": (1, 0, 0, 0),
    "A5": (1, 0, 0, 0),
    "A6": (1, 0, 0, 0),
    "B0": (0.707, -0.707, 0, 0),
    "B1": (0.707, -0.707, 0, 0),
    "B2": (0.707, -0.707, 0, 0),
    "B3": (0.707, 0, -0.707, 0),
    "B4": (0.707, -0.707, 0, 0),
    "B5": (0.707, -0.707, 0, 0),
    "B6": (0.707, 0.707, 0, 0),
    "D6": (0.707, -0.707, 0, 0),
    "E6": (0.707, 0, 0.707, 0),
    "C1": (0.707, 0.707, 0, 0),
    "C3": (0, -1, 0, 0),
    "E2": (0.707, 0.707, 0, 0),
    "E3": (0.707, 0.707, 0, 0),
    "E4": (0.707, 0.707, 0, 0),
    "E5": (0.707, 0.707, 0, 0),
    "F1": (0.707, 0.707, 0, 0),
    "F3": (0.707, 0, 0.707, 0),
    "F4": (0.707, 0.707, 0, 0),
}

EGAD_RESCALE = {
    "A0":
    [0.0009450984919849741, 0.0009450984919849741, 0.0009450984919849741],
    "A1":
    [0.000864752231683409, 0.000864752231683409, 0.000864752231683409 * 2],
    "A6":
    [0.000808513314869247, 0.000808513314869247, 0.000808513314869247 * 2],
    "A5":
    [0.0008905764542860835, 0.0008905764542860835, 0.0008905764542860835 * 2],
    "B0":
    [0.0008744435954728685, 0.0008744435954728685 * 2, 0.0008744435954728685],
    "B1":
    [0.0008039916839692638, 0.0008039916839692638 * 2, 0.0008039916839692638],
    "B4":
    [0.0008007987965898657, 0.0008007987965898657 * 2, 0.0008007987965898657],
    "B5":
    [0.0008860371509720841, 0.0008860371509720841 * 2, 0.0008860371509720841]
    # "B0": (0.77, 0.77, 0, 0),
    # "B2": (0.77, 0.77, 0., 0),
    # "B1": (0.77, 0.77, 0, 0),
    # "B3": (0.77, 0, 0.77, 0),
    # "B6": (0.77, 0.77, 0, 0),
    # "C1": (0.77, 0.77, 0, 0),
}

EGAD_KIND = {
    "A": ["A0", "A1", "A2", "A3", "A4", "A5", "A6"],
    "B": ["B0", "B1", "B2", "B3", "B4", "B5", "B6"],
    "C": ["C0", "C1", "C2", "C3", "C4", "C5", "C6"],
    "D": ["D0", "D1", "D2", "D3", "D4", "D5", "D6"],
    "E": ["E1", "E2", "E3", "E4", "E5", "E6"],
    "F": ["F1", "F2", "F3", "F4", "F5", "F6"],
    "G": ["G2", "G3", "G4", "G5", "G6"]
}
#============== add orientation===================


def get_egad_root_dir():
    current_dir = Path(__file__).parent
    ycb_dir = current_dir.parent.parent / "assets" / "egad"
    return ycb_dir.resolve()


def load_egad_scale():
    eval_file = get_egad_root_dir() / "info_eval_v0.json"
    with eval_file.open() as f:
        egad_scale = {"eval": json.load(f)}
    return egad_scale


def load_egad_name():
    egad_dir = get_egad_root_dir()
    entities = {"train": "egad_train_set"}
    exclude = {
        "train":
        ["E0", "E2", "F0", "G0", "G1", "F6", 'G6', 'G5', 'G4', 'G3', 'G2']
    }
    name_dict = {}
    for split, sub_dir in entities.items():
        name_dict[split] = list()
        for file in (egad_dir / sub_dir).glob("*.obj"):
            if file.stem not in exclude[split]:
                name_dict[split].append(file.stem)

    return name_dict


EGAD_SCALE = load_egad_scale()
EGAD_NAME = load_egad_name()


def load_egad_object(scene: sapien.Scene,
                     model_id: str,
                     physical_material: sapien.PhysicalMaterial = None,
                     density=1000,
                     visual_only=False):
    # Source: https://github.com/haosulab/ManiSkill2022/tree/main/scripts/jigu/egad
    builder = scene.create_actor_builder()
    # A heuristic way to infer split
    split = "train" if "_" in model_id else "eval"
    # if split == "eval":
    #     scale = EGAD_SCALE[split][model_id]["scales"]
    # else:
    #     raise NotImplementedError
    scale = EGAD_SCALE[split][model_id]["scales"]

    if physical_material is None:
        physical_material = scene.engine.create_physical_material(1, 1, 0.01)
    egad_dir = get_egad_root_dir()
    scale_factor = 1.5

    scales = np.array(scale * 3) / scale_factor

    # if model_id in EGAD_RESCALE.keys():
    #     scales = EGAD_RESCALE[model_id]
    # EGAD_ORIENTATION[model_id] = [1, 0, 0, 0]

    # if model_id not in EGAD_ORIENTATION.keys():
    #     EGAD_ORIENTATION[model_id] = [1, 0, 0, 0]
    if not visual_only:
        collision_file = str(egad_dir / "egad_{split}_set_vhacd" /
                             f"{model_id}.obj").format(split=split)
        builder.add_multiple_collisions_from_file(
            filename=collision_file,
            scale=scales,
            material=physical_material,
            density=density,
        )

    visual_file = str(egad_dir / "egad_{split}_set" /
                      f"{model_id}.obj").format(split=split)
    builder.add_visual_from_file(
        filename=visual_file,
        scale=scales,
    )

    if not visual_only:
        actor = builder.build(name=model_id)
    else:
        actor = builder.build_static(name=model_id)
    return actor, scales
