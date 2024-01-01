import json
from pathlib import Path
from xml.parsers.expat import model

import numpy as np
import sapien.core as sapien
from scipy.spatial.transform import Rotation as R


def build_static(scene, model_type, model_name, scale):
    builder = scene.create_actor_builder()

    visual_file = "./assets/partnet-mobility-dataset/" + model_type + "/visual/" + model_name + "/" + model_name + ".stl"

    pose = [1, 0, 0, 0]
    builder.add_visual_from_file(filename=visual_file,
                                 scale=np.array([scale] * 3),
                                 pose=sapien.Pose(q=np.array(pose)))

    actor = builder.build_static(name=model_name)

    return actor


def load_nut_bolt_object(scene: sapien.Scene,
                         model_type: str,
                         model_name: str,
                         physical_material: sapien.PhysicalMaterial = None,
                         density=1000,
                         visual_only=False,
                         random=False,
                         scale=None):

    filename = "./assets/partnet-mobility-dataset/bolt_nut/" + model_name + ".urdf"
    loader = scene.create_urdf_loader()

    loader.scale = 0.2
    height = 0.15

    material = scene.create_physical_material(1, 1, 0)

    config = {'material': material, 'density': 1000}
    builder = loader.load_file_as_articulation_builder(filename, config)

    for link in builder.get_link_builders():
        link.set_collision_groups(1, 1, 4, 4)

    if not visual_only:
        loader.load_multiple_collisions_from_file = True

        object = builder.build(fix_root_link=True)
    else:

        object = build_static(scene, model_type, model_name, scale)

        return None, height, loader.scale

    return object, height, loader.scale
