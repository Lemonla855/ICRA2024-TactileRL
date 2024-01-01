from http.client import PARTIAL_CONTENT
import json
from pathlib import Path
from xml.parsers.expat import model

import numpy as np
import sapien.core as sapien
from scipy.spatial.transform import Rotation as R
import pdb
import trimesh
from hand_teleop.utils.partnet_class_utils import modify_scale, bottle3_dict


def get_assembling_kits_root_dir():
    current_dir = Path(__file__).parent
    shapenet_dir = current_dir.parent.parent / "assets" / "assembling_kits"
    return shapenet_dir.resolve()


def load_assembling_kits_object(
        scene: sapien.Scene,
        cat_id: str,
        model_id: str,
        physical_material: sapien.PhysicalMaterial = None,
        density=1000,
        visual_only=False):
    builder = scene.create_articulation_builder()

    r = R.from_euler('xyz', [0, -90, 180], degrees=True)
    orientation = r.as_quat()[[3, 0, 1, 2]]  #x,y,z,w
    # position = np.array([0, 0., 0.07])
    # orientation = [1, 0, 0, 0]

    # link_alpha
    link_alpha = builder.create_link_builder()
    link_alpha.set_name("link_alpha")
    scales = np.array([0.015, 0.015, 0.004]) / 1.5
    link_alpha.add_multiple_collisions_from_file(
        "assets/assembling_kits/models/collision/" + model_id + ".obj",
        pose=sapien.Pose([0, 0, 0], orientation),
        scale=scales,
        density=1000)
    link_alpha.add_visual_from_file("assets/assembling_kits/models/visual/" +
                                    model_id + ".obj",
                                    pose=sapien.Pose([0, 0, 0], orientation),
                                    scale=scales)

    # link handle
    link_handle = builder.create_link_builder()
    link_alpha.set_joint_properties(joint_type="revolute",
                                    limits=[[-2 * np.pi, 2 * np.pi]],
                                    pose_in_child=sapien.Pose([0, 0, -0.1],
                                                              [1, 0, 0, 0]),
                                    pose_in_parent=sapien.Pose([0, 0, -0.1],
                                                               [1, 0, 0, 0]))
    link_alpha.set_parent(link_handle.get_index())

    height = 0.1
    for link in builder.get_link_builders():
        link.set_collision_groups(1, 1, 4, 4)
    actor = builder.build(fix_root_link=True)

    return actor, height


def load_partnet_object(scene: sapien.Scene,
                        model_type: str,
                        model_name: str,
                        physical_material: sapien.PhysicalMaterial = None,
                        density=1000,
                        visual_only=False,
                        random=False,
                        scale=None,
                        renderer=None,
                        down=False):

    height = 0.2

    builder: sapien.ArticulationBuilder = scene.create_articulation_builder()

    bottle_scales, cap_scales, y_length_bottle, y_length_cap = modify_scale(
        bottle3_dict[model_name]["link"], scale)

    # link_0
    link_cap = builder.create_link_builder()
    link_cap.set_name("link_0_0")
    scales = cap_scales
    link_cap.add_multiple_collisions_from_file(
        "./assets/partnet-mobility-dataset/bottle/collision/" + model_name +
        "/link_0_0.stl",
        pose=sapien.Pose([0, 0, 0], [0.707, 0.707, 0, 0]),
        scale=scales,
        density=1000)
    link_cap.add_visual_from_file(
        "./assets/partnet-mobility-dataset/bottle/collision/" + model_name +
        "/link_0_0.stl",
        pose=sapien.Pose([0, 0, 0], [0.707, 0.707, 0, 0]),
        scale=scales)

    #link_1
    link_bottle = builder.create_link_builder()
    link_bottle.set_name("link_1")
    scales = bottle_scales
    link_bottle.add_multiple_collisions_from_file(
        "./assets/partnet-mobility-dataset/bottle/collision/" + model_name +
        "/link_1.stl",
        pose=sapien.Pose([0, 0, 0], [0.707, 0.707, 0, 0]),
        scale=scales,
        density=1000)
    link_bottle.add_visual_from_file(
        "./assets/partnet-mobility-dataset/bottle/collision/" + model_name +
        "/link_1.stl",
        pose=sapien.Pose([0, 0, 0], [0.707, 0.707, 0, 0]),
        scale=scales)

    link_cap.set_joint_properties(joint_type="fixed",
                                  limits=[],
                                  pose_in_parent=sapien.Pose([0, 0, 0],
                                                             [1, 0, 0, 0]))
    link_cap.set_parent(link_bottle.get_index())

    #
    link_handle = builder.create_link_builder()
    link_bottle.set_joint_properties(joint_type="revolute",
                                     limits=[[-2 * np.pi, 2 * np.pi]],
                                     pose_in_child=sapien.Pose([0, 0, -0.1],
                                                               [1, 0, 0, 0]),
                                     pose_in_parent=sapien.Pose([0, 0, -0.1],
                                                                [1, 0, 0, 0]))
    link_bottle.set_parent(link_handle.get_index())

    for link in builder.get_link_builders():
        link.set_collision_groups(1, 1, 4, 4)
    object = builder.build(fix_root_link=True)

    return object, height, scale
