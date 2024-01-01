import json
from pathlib import Path

import numpy as np
import sapien.core as sapien

import pdb

MODELNET40_SCALE = {
    "vase_0458": 0.002,
    "vase_0305": 0.001,
    "vase_0113": 0.03,
    "vase_0202": 0.01,
    "vase_0268": 0.01,
    "vase_0418": 0.0008,
    "vase_0441": 0.01,
    "vase_0221": 0.01,
    "vase_0452": 0.002,
    "vase_0210": 0.01,
    "vase_0063": 0.01,  # not good
    "vase_0283": 0.003,
    "vase_0109": 0.002,
    "vase_0144": 0.00008,
    "vase_0154": 0.008,
    "vase_0343": 0.004,
    "vase_0068": 0.01,
    "vase_0292": 0.01,
    "vase_0129": 0.005,
    "vase_0246": 0.005,
    "vase_0114": 0.005,
    "bottle_0188": 0.02,
    "bottle_0162": 0.025,
    "bottle_0271": 0.02,
    "bottle_0130": 0.0005,
    "bottle_0215": 0.01,
    "bottle_0041": 0.02,  # less pattern
    "bottle_0187": 0.01,
    "bottle_0074": 0.0003,  # need align
    "bottle_0042": 0.0003
}

MODELNET40_ANYTRAIN = [
    "vase_0305", "vase_0210", "vase_0154", "vase_0343", "vase_0068",
    "vase_0129", "vase_0246", "vase_0114"
]

MODELNET40_SCALE_INFO = {
    "vase": {
        "vase_0305": {
            "scale": 0.0010690813651099951,
            "height": 187.0765
        },
        "vase_0210": {
            "scale": 0.011894137183528974,
            "height": 16.815007
        },
        "vase_0154": {
            "scale": 0.004946431470346566,
            "height": 40.433189299999995
        },
        "vase_0343": {
            "scale": 0.004819873959813964,
            "height": 41.494861
        },
        "vase_0068": {
            "scale": 0.010553811245085882,
            "height": 18.9505
        },
        "vase_0129": {
            "scale": 0.004946983805299046,
            "height": 40.4286749
        },
        "vase_0246": {
            "scale": 0.0031135395037929364,
            "height": 64.2355749
        },
        "vase_0114": {
            "scale": 0.0031135395037929364,
            "height": 64.2355749
        }
    }
}


def get_modelnet_root_dir():
    current_dir = Path(__file__).parent
    modelnet_dir = current_dir.parent.parent / "assets" / "modelnet"
    # info_path = modelnet_dir / "info.json"
    # with info_path.open("r") as f:
    #     cat_scale = json.load(f)
    cat_scale = MODELNET40_SCALE_INFO
    return modelnet_dir.resolve(), cat_scale


def load_modelnet_object(scene: sapien.Scene,
                         model_name: str,
                         physical_material: sapien.PhysicalMaterial = None,
                         density=1000,
                         visual_only=False):

    builder = scene.create_actor_builder()
    # A heuristic way to infer split
    modelnet_dir, cat_scale = get_modelnet_root_dir()
    if model_name in MODELNET40_SCALE.keys():
        scales = np.array(
            [cat_scale[model_name.split("_")[0]][model_name]["scale"]] * 3)
    else:
        raise NotImplementedError

    if not visual_only:
        collision_name = str(model_name.split("_")[0]) + "_vhac"

        collision_file = str(modelnet_dir / collision_name /
                             f"{model_name}.obj")

        builder.add_multiple_collisions_from_file(filename=collision_file,
                                                  scale=scales,
                                                  material=physical_material,
                                                  density=density,
                                                  pose=sapien.Pose())

    visual_name = str(model_name.split("_")[0]) + "_visual"
    visual_file = str(modelnet_dir / visual_name / f"{model_name}.obj")
    builder.add_visual_from_file(filename=visual_file,
                                 scale=scales,
                                 pose=sapien.Pose())

    if not visual_only:
        actor = builder.build(name=model_name)
    else:
        actor = builder.build_static(name=model_name)

    return actor, cat_scale[model_name.split(
        "_")[0]][model_name]["height"] * 0.5 * cat_scale[model_name.split(
            "_")[0]][model_name]["scale"]
