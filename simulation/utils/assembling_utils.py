import json
from pathlib import Path

import numpy as np
import sapien.core as sapien
import pdb
from hand_teleop.utils.partnet_utils import load_mesh


def get_assembling_kits_root_dir():
    current_dir = Path(__file__).parent
    shapenet_dir = current_dir.parent.parent / "assets" / "assembling_kits"
    return shapenet_dir.resolve()


def load_assembling_kits_object(
        scene: sapien.Scene,
        model_id: str,
        physical_material: sapien.PhysicalMaterial = None,
        density=1000,
        visual_only=False):
    builder = scene.create_actor_builder()

    if physical_material is None:
        physical_material = scene.engine.create_physical_material(1.5, 1, 0.01)

    collision_file = "assets/partnet-mobility-dataset/assembling_kits/collision/" + model_id + ".obj"
    visual_file = "assets/partnet-mobility-dataset/assembling_kits/visual/" + model_id + ".obj"

    x_length, y_length, z_length, x_center, y_center, z_center = load_mesh(
        "assets/partnet-mobility-dataset/assembling_kits/visual/" + model_id +
        ".obj")

    scales = np.array([0.08 / x_length, 0.12 / y_length, 0.035 / z_length])

    # from scipy.spatial.transform import Rotation as R
    # r = R.from_rotvec([np.pi / 2, -np.pi / 2, 0])
    # quat = r.as_quat()[[3, 0, 1, 2]]  #x,y,z,w
    quat = np.array([1, 0, 0, 0])

    if not visual_only:
        builder.add_multiple_collisions_from_file(filename=collision_file,
                                                  scale=scales,
                                                  material=physical_material,
                                                  density=density,
                                                  pose=sapien.Pose(q=quat))

    builder.add_visual_from_file(filename=visual_file,
                                 scale=scales,
                                 pose=sapien.Pose(q=quat))

    if not visual_only:
        actor = builder.build(name=model_id)
    else:
        actor = builder.build_static(name=model_id)

    return actor, scales, abs(y_center - y_length / 2) * scales[1], np.array(
        [x_length, y_length, z_length]) * np.array(scales) / 2
