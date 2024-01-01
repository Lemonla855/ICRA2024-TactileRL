import json
from pathlib import Path

import numpy as np
import sapien.core as sapien
import pdb

from hand_teleop.utils.partnet_utils import load_mesh

import os


def load_bolt_name():
    list_names = os.listdir("./assets/bolt/collision")
    bolt_name = []

    for name in list_names:

        bolt_name.append(name.split(".")[0])

    return bolt_name


BOLT_NAMES = load_bolt_name()

SHAPENET_CAT = [
    "02876657",
    "02946921",  #"04460130", "2801938", "02747177", "03991062"
]

SHAPNET_SCALE_INFO = {
    "02747177": {
        "ffe5f0ef45769204cb2a965e75be701c": {
            "height": 0.06709580449514238,
            "scales": [0.17615116998244254]
        },
        "ccd313055ecb8799f155d75bbf62b80": {
            "height": 0.06709580449514238,
            "scales": [0.17615116998244254]
        },
        "41b946cebf6b76cb25d3c528e036a532": {
            "height": 0.06709580449514238,
            "scales": [0.17615116998244254]
        },
        "4dbbece412ef64b6d2b12aa6a0f050b3": {
            "height": 0.06709580449514238,
            "scales": [0.17615116998244254]
        },
        "9b5eeb473448827ff155d75bbf62b80": {
            "height": 0.06709580449514238,
            "scales": [0.17615116998244254]
        },
        "31d318b5f2a1be9ee88c1d6fc3580355": {
            "height": 0.06709580449514238,
            "scales": [0.17615116998244254]
        },
        "415cae243e110f5cf155d75bbf62b80": {
            "height": 0.06709580449514238,
            "scales": [0.17615116998244254]
        },
        "89aff0d006fc22ff9405d3391cbdb79b": {
            "height": 0.06709580449514238,
            "scales": [0.17615116998244254]
        },
        "ea5598f84391296ad2b12aa6a0f050b3": {
            "height": 0.06709580449514238,
            "scales": [0.17615116998244254]
        },
        "8be2cf5a52644885cb2a965e75be701c": {
            "height": 0.06709580449514238,
            "scales": [0.17615116998244254]
        },
        "269b2573816e52dad2b12aa6a0f050b3": {
            "height": 0.06709580449514238,
            "scales": [0.17615116998244254]
        },
        "9ee464a3fb9d3e8e57cd6640bbeb736d": {
            "height": 0.06709580449514238,
            "scales": [0.17615116998244254]
        },
        "2f06f66740e65248ba6cf6cbb9f4c2bb": {
            "height": 0.06709580449514238,
            "scales": [0.17615116998244254]
        },
        "95c5a762fc8624859d7f638b2b2e0564": {
            "height": 0.06709580449514238,
            "scales": [0.17615116998244254]
        },
        "5c7bd882d399e031d2b12aa6a0f050b3": {
            "height": 0.06709580449514238,
            "scales": [0.17615116998244254]
        },
        "f10ced1e1dcf4f32f155d75bbf62b80": {
            "height": 0.06709580449514238,
            "scales": [0.17615116998244254]
        },
        "81a04a676ff96a94b93fc2b66c6b86b6": {
            "height": 0.06709580449514238,
            "scales": [0.17615116998244254]
        },
        "45d71fc0e59af8bf155d75bbf62b80": {
            "height": 0.06709580449514238,
            "scales": [0.17615116998244254]
        },
        "97e3b7b4688c2a93f155d75bbf62b80": {
            "height": 0.06709580449514238,
            "scales": [0.17615116998244254]
        },
        "eddce90cae7f1468f155d75bbf62b80": {
            "height": 0.06709580449514238,
            "scales": [0.17615116998244254]
        },
        "b689aed9b7017c49f155d75bbf62b80": {
            "height": 0.06709580449514238,
            "scales": [0.17615116998244254]
        },
        "b537facfbab18da781faebbdea6bd9be": {
            "height": 0.06709580449514238,
            "scales": [0.17615116998244254]
        },
        "5d756e92b734f2d2b12aa6a0f050b3": {
            "height": 0.06709580449514238,
            "scales": [0.17615116998244254]
        },
        "def815f84e0cc9cfcb2a965e75be701c": {
            "height": 0.06709580449514238,
            "scales": [0.17615116998244254]
        },
        "6d1aacdd49c4bac781faebbdea6bd9be": {
            "height": 0.06709580449514238,
            "scales": [0.17615116998244254]
        },
        "21a94be58900df5bcb2a965e75be701c": {
            "height": 0.06709580449514238,
            "scales": [0.17615116998244254]
        },
        "65d2760c534966f2d2b12aa6a0f050b3": {
            "height": 0.06709580449514238,
            "scales": [0.17615116998244254]
        },
        "af7a781f08fdd4481faebbdea6bd9be": {
            "height": 0.06709580449514238,
            "scales": [0.17615116998244254]
        },
        "49bff2b8bcdc50dfba6cf6cbb9f4c2bb": {
            "height": 0.06709580449514238,
            "scales": [0.17615116998244254]
        },
        "fe2b3c0ca29baeaed2b12aa6a0f050b3": {
            "height": 0.06709580449514238,
            "scales": [0.17615116998244254]
        },
        "ec1c1aa7003cf68d49e6f7df978f3373": {
            "height": 0.06709580449514238,
            "scales": [0.17615116998244254]
        },
        "91d6f4726d1a169d924bf081da6f024c": {
            "height": 0.06709580449514238,
            "scales": [0.17615116998244254]
        },
        "7a931ec996edbfeacb2a965e75be701c": {
            "height": 0.06709580449514238,
            "scales": [0.17615116998244254]
        },
        "1f4f53bc04eefca6f155d75bbf62b80": {
            "height": 0.06709580449514238,
            "scales": [0.17615116998244254]
        },
        "e6ea9e633efbe1e281faebbdea6bd9be": {
            "height": 0.06709580449514238,
            "scales": [0.17615116998244254]
        },
        "156fe84679171501f155d75bbf62b80": {
            "height": 0.06709580449514238,
            "scales": [0.17615116998244254]
        },
        "f8b449eee1c8775fd2b12aa6a0f050b3": {
            "height": 0.06709580449514238,
            "scales": [0.17615116998244254]
        },
        "7c90fba6cd7f73871c1ef519b9196b63": {
            "height": 0.06709580449514238,
            "scales": [0.17615116998244254]
        },
        "b6833a66eedf457af155d75bbf62b80": {
            "height": 0.06709580449514238,
            "scales": [0.17615116998244254]
        },
        "f53492ed7a071e13cb2a965e75be701c": {
            "height": 0.06709580449514238,
            "scales": [0.17615116998244254]
        },
        "d7394e554aff20b2f155d75bbf62b80": {
            "height": 0.06709580449514238,
            "scales": [0.17615116998244254]
        },
        "8a321a0750972afff155d75bbf62b80": {
            "height": 0.06709580449514238,
            "scales": [0.17615116998244254]
        },
        "16ce4a2ff48b8a81d2b12aa6a0f050b3": {
            "height": 0.06709580449514238,
            "scales": [0.17615116998244254]
        },
        "22213dc6019c215a81faebbdea6bd9be": {
            "height": 0.06709580449514238,
            "scales": [0.17615116998244254]
        },
        "64393896dc4403c0e88c1d6fc3580355": {
            "height": 0.06709580449514238,
            "scales": [0.17615116998244254]
        },
        "92fa62263ad30506d2b12aa6a0f050b3": {
            "height": 0.06709580449514238,
            "scales": [0.17615116998244254]
        },
        "f76b14ffc2fe6c7485eb407ec5a0fadd": {
            "height": 0.06709580449514238,
            "scales": [0.17615116998244254]
        },
        "8b0728eb28c1873938f21a3304cc4bdc": {
            "height": 0.06709580449514238,
            "scales": [0.17615116998244254]
        },
        "ada5b924669c5bf4cb2a965e75be701c": {
            "height": 0.06709580449514238,
            "scales": [0.17615116998244254]
        },
        "b9b1108520918039a47fdd008f2ae822": {
            "height": 0.06709580449514238,
            "scales": [0.17615116998244254]
        },
        "2f79bca58f58a3ead2b12aa6a0f050b3": {
            "height": 0.06709580449514238,
            "scales": [0.17615116998244254]
        },
        "78f8bb00a5850aaf81faebbdea6bd9be": {
            "height": 0.06709580449514238,
            "scales": [0.17615116998244254]
        },
        "18be3d380d248fe081faebbdea6bd9be": {
            "height": 0.06709580449514238,
            "scales": [0.17615116998244254]
        },
        "4117be347627a845ba6cf6cbb9f4c2bb": {
            "height": 0.06709580449514238,
            "scales": [0.17615116998244254]
        },
        "1eb3abf47be2022381faebbdea6bd9be": {
            "height": 0.06709580449514238,
            "scales": [0.17615116998244254]
        },
        "ded74a76b1a4dc8ecb2a965e75be701c": {
            "height": 0.06709580449514238,
            "scales": [0.17615116998244254]
        },
        "71f3b101a6d47811cb2a965e75be701c": {
            "height": 0.06709580449514238,
            "scales": [0.17615116998244254]
        },
        "f2c7e1b8112282dbcb2a965e75be701c": {
            "height": 0.06709580449514238,
            "scales": [0.17615116998244254]
        },
        "23a38f8cafa5bc9657a8fa4c1cbcf3ea": {
            "height": 0.06709580449514238,
            "scales": [0.17615116998244254]
        },
        "af6fa396b2869446d4d8765e3910f617": {
            "height": 0.06709580449514238,
            "scales": [0.17615116998244254]
        },
        "46fca3bdcd454402f155d75bbf62b80": {
            "height": 0.06709580449514238,
            "scales": [0.17615116998244254]
        },
        "5eea24527c8b3124cb2a965e75be701c": {
            "height": 0.06709580449514238,
            "scales": [0.17615116998244254]
        },
        "b729214e49af571bf155d75bbf62b80": {
            "height": 0.06709580449514238,
            "scales": [0.17615116998244254]
        },
        "33e36da1afca5e57f155d75bbf62b80": {
            "height": 0.06709580449514238,
            "scales": [0.17615116998244254]
        },
        "76bd9785d3f6e77f81faebbdea6bd9be": {
            "height": 0.06709580449514238,
            "scales": [0.17615116998244254]
        },
        "a8b39c32604173c1d2b12aa6a0f050b3": {
            "height": 0.06709580449514238,
            "scales": [0.17615116998244254]
        },
        "a23c4789341aa242cb2a965e75be701c": {
            "height": 0.06709580449514238,
            "scales": [0.17615116998244254]
        },
        "8fff3a4ba8db098bd2b12aa6a0f050b3": {
            "height": 0.06709580449514238,
            "scales": [0.17615116998244254]
        },
        "555a9bff25db49ddcb2a965e75be701c": {
            "height": 0.06709580449514238,
            "scales": [0.17615116998244254]
        },
        "4606c9bae0c3a717ba6cf6cbb9f4c2bb": {
            "height": 0.06709580449514238,
            "scales": [0.17615116998244254]
        },
        "c0686f7b0da06f36cb2a965e75be701c": {
            "height": 0.06709580449514238,
            "scales": [0.17615116998244254]
        },
        "1532546ac3d8a2cff155d75bbf62b80": {
            "height": 0.06709580449514238,
            "scales": [0.17615116998244254]
        },
        "e5050955cb29144b907bde230eb962dd": {
            "height": 0.06709580449514238,
            "scales": [0.17615116998244254]
        },
        "e91da0c5c59ec4c8cb2a965e75be701c": {
            "height": 0.06709580449514238,
            "scales": [0.17615116998244254]
        },
        "663082bdb695ea9ccb2a965e75be701c": {
            "height": 0.06709580449514238,
            "scales": [0.17615116998244254]
        },
        "af1dc226f465990e81faebbdea6bd9be": {
            "height": 0.06709580449514238,
            "scales": [0.17615116998244254]
        },
        "ae33867e3c5e1703f155d75bbf62b80": {
            "height": 0.06709580449514238,
            "scales": [0.17615116998244254]
        },
        "fc7ba0ce66b9dfcff155d75bbf62b80": {
            "height": 0.06709580449514238,
            "scales": [0.17615116998244254]
        },
        "c50c72eefe225b51cb2a965e75be701c": {
            "height": 0.06709580449514238,
            "scales": [0.17615116998244254]
        },
        "b51812771e42354f9996a93ae0c9395c": {
            "height": 0.06709580449514238,
            "scales": [0.17615116998244254]
        },
        "d756ab06172e422fa1228be00ccc1d69": {
            "height": 0.06709580449514238,
            "scales": [0.17615116998244254]
        },
        "2a20ec86e1aaf196cb2a965e75be701c": {
            "height": 0.06709580449514238,
            "scales": [0.17615116998244254]
        },
        "66d96d2428184442ba6cf6cbb9f4c2bb": {
            "height": 0.06709580449514238,
            "scales": [0.17615116998244254]
        },
        "24884ef01e9a3832d2b12aa6a0f050b3": {
            "height": 0.06709580449514238,
            "scales": [0.17615116998244254]
        }
    },
}

SHAPENET_BBPoint = {
    "02946921": {
        'fd40fa8939f5f832ae1aa888dd691e79':
        np.array([-0.207186, -0.207186, -0.407942]),
        '3a7d8f866de1890bab97e834e9ba876c':
        np.array([-0.207319, -0.207319, -0.407853]),
        '5505ddb926a77c0f171374ea58140325':
        np.array([-0.227403, -0.227402, -0.388253]),
        'd801d5b05f7d872341d8650f1ad40eb1':
        np.array([-0.233035, -0.233035, -0.381612]),
        '3a9041d7aa0b4b9ad9802f6365035049':
        np.array([-0.241852, -0.241852, -0.370584]),
        '90d40359197c648b23e7e4bd2944793':
        np.array([-0.218987, -0.218944, -0.397649]),
        '6b2c6961ad0891936193d9e76bb15876':
        np.array([-0.242918, -0.242918, -0.36921]),
        'e706cf452c4c124d77335fb90343dc9e':
        np.array([-0.214394, -0.214356, -0.402534]),
        'bf974687b678c66e93fb5c975e8de2b7':
        np.array([-0.209119, -0.209119, -0.407944]),
        '5bd768cde93ec1acabe235874aea9b9b':
        np.array([-0.221614, -0.221614, -0.39469]),
        'ac66fb0ff0d50368ced499bff9a86355':
        np.array([-0.221309, -0.221289, -0.395016]),
        '60f4012b5336902b30612f5c0ef21eb8':
        np.array([-0.213463, -0.213464, -0.403504]),
        '6a703fd2b09f50f347df6165146d5bbd':
        np.array([-0.21003, -0.210032, -0.407055]),
        '29bc4b2e86b91d392e06d87a0fadf00':
        np.array([-0.229141, -0.229139, -0.383797]),
        '669033f9b748c292d18ceeb5427760e8':
        np.array([-0.231162, -0.231162, -0.383761]),
        '97ca02ee1e7b8efb6193d9e76bb15876':
        np.array([-0.242563, -0.242563, -0.369611]),
        '408028e3bdd8d05b2d6c8e51365a5a87':
        np.array([-0.222914, -0.222914, -0.393289]),
        'fd73199d9e01927fffc14964b2380c14':
        np.array([-0.213474, -0.213474, -0.403551]),
        'c4bce3dc44c66630282f9cd3f45eaa2a':
        np.array([-0.18419, -0.18419, -0.43088]),
        'baaa4b9538caa7f06e20028ed3cb196e':
        np.array([-0.165199, -0.165167, -0.445707]),
        '990a058fbb51c655d773a8448a79e14c':
        np.array([-0.224484, -0.224485, -0.391418]),
        'e8c446c352b84189bc8d62498ee8e85f':
        np.array([-0.22596, -0.22596, -0.389865]),
        '17ef524ca4e382dd9d2ad28276314523':
        np.array([-0.227356, -0.226998, -0.386543]),
        '788094fbf1a523f768104c9df46104ca':
        np.array([-0.236658, -0.236658, -0.377151]),
        'ace554442d7d530830612f5c0ef21eb8':
        np.array([-0.244483, -0.244483, -0.367113]),
        '3c8af6b0aeaf13c2abf4b6b757f4f768':
        np.array([-0.217707, -0.21771, -0.398888])
    }
}


def get_shapenet_root_dir():
    current_dir = Path(__file__).parent
    shapenet_dir = current_dir.parent.parent / "assets" / "shapenet"
    return shapenet_dir.resolve()


def load_shapenet_object_list():
    cat_dict = {}
    info_path = get_shapenet_root_dir() / "info.json"
    with info_path.open("r") as f:
        cat_scale = json.load(f)

    for cat in SHAPENET_CAT:
        object_list_file = get_shapenet_root_dir() / f"{cat}.txt"
        with object_list_file.open("r") as f:
            cat_object_list = f.read().split("\n")
        cat_dict[cat] = {}

        for model_id in cat_object_list[:-1]:

            cat_dict[cat][model_id] = cat_scale[cat][model_id]
        cat_dict[cat]["train"] = cat_object_list[:10]
        cat_dict[cat]["eval"] = cat_object_list[10:]

    # cat_dict=SHAPNET_SCALE_INFO
    return cat_dict


CAT_DICT = load_shapenet_object_list()


def load_shapenet_object(scene: sapien.Scene,
                         cat_id: str,
                         model_id: str,
                         physical_material: sapien.PhysicalMaterial = None,
                         density=1000,
                         visual_only=False,
                         scale_factor=2.0,
                         height_factor=1.5):
    builder = scene.create_actor_builder()

    if physical_material is None:
        physical_material = scene.engine.create_physical_material(1.5, 1, 0.01)
    shapenet_dir = get_shapenet_root_dir()
    collision_file = str(shapenet_dir / cat_id / model_id / "convex.obj")
    visual_file = str(shapenet_dir / cat_id / model_id / "model.obj")

    info = CAT_DICT[cat_id][model_id]
    scale = info["scales"]
    height = info["height"]

    scales = np.array(scale * 3)
    scales[[0, 2]] /= scale_factor
    scales[1] /= height_factor

    x_length, y_length, z_length, x_center, y_center, z_center = load_mesh(
        collision_file)

    if not visual_only:
        builder.add_multiple_collisions_from_file(
            filename=collision_file,
            scale=scales,
            material=physical_material,
            density=density,
            pose=sapien.Pose(q=np.array([0.7071, 0.7071, 0, 0])))

    builder.add_visual_from_file(
        filename=visual_file,
        scale=scales,
        pose=sapien.Pose(q=np.array([0.7071, 0.7071, 0, 0])))

    if not visual_only:
        actor = builder.build(name=model_id)
    else:
        actor = builder.build_static(name=model_id)

    return actor, abs(y_center -
                      abs(y_length) / 2) * scales[1], scales, np.array([
                          x_length, y_length, z_length
                      ]) * np.array(scales) / 2


def load_bolt_object(scene: sapien.Scene,
                     cat_id: str,
                     model_id: str,
                     physical_material: sapien.PhysicalMaterial = None,
                     density=1000,
                     visual_only=False,
                     scale_factor=2.0,
                     height_factor=1.5):
    builder = scene.create_actor_builder()

    if physical_material is None:
        physical_material = scene.engine.create_physical_material(1.5, 1, 0.01)

    collision_file = "./assets/bolt/collision/" + model_id + ".stl"
    visual_file = "./assets/bolt/visual/" + model_id + ".stl"

    x_length, y_length, z_length, x_center, y_center, z_center = load_mesh(
        collision_file)

    scale = 0.04 / x_length
    scales = np.array([0.08 / x_length, 0.08 / y_length, 0.12 / z_length])

    if not visual_only:
        builder.add_multiple_collisions_from_file(
            filename=collision_file,
            scale=scales,
            material=physical_material,
            density=density,
            pose=sapien.Pose(q=np.array([1, 0, 0, 0])))

    builder.add_visual_from_file(filename=visual_file,
                                 scale=scales,
                                 pose=sapien.Pose(q=np.array([1, 0, 0, 0])))

    if not visual_only:
        actor = builder.build(name=model_id)
    else:
        actor = builder.build_static(name=model_id)

    return actor, abs(z_center -
                      abs(z_length) / 2) * scales[2], scales, np.array([
                          x_length, y_length, z_length
                      ]) * np.array(scales) / 2


def load_shapenet_articulate_object(
        scene: sapien.Scene,
        cat_id: str,
        model_id: str,
        physical_material: sapien.PhysicalMaterial = None,
        density=1000,
        visual_only=False):

    if physical_material is None:
        physical_material = scene.engine.create_physical_material(1.5, 1, 0.01)
    shapenet_dir = get_shapenet_root_dir()
    collision_file = str(shapenet_dir / cat_id / model_id / "convex.obj")
    visual_file = str(shapenet_dir / cat_id / model_id / "model.obj")

    info = CAT_DICT[cat_id][model_id]
    scale = info["scales"]
    height = info["height"] / 1.2

    builder: sapien.ArticulationBuilder = scene.create_articulation_builder()
    obj_link = builder.create_link_builder()

    obj_link.add_multiple_collisions_from_file(
        collision_file,
        scale=np.array(scale * 3) / 1.2,
        pose=sapien.Pose(q=np.array([0.7071, 0.7071, 0, 0])))
    obj_link.add_visual_from_file(
        visual_file,
        scale=np.array(scale * 3) / 1.2,
        pose=sapien.Pose(q=np.array([0.7071, 0.7071, 0, 0])))

    slider_link = builder.create_link_builder()

    bbox_points = SHAPENET_BBPoint[cat_id][model_id] * scale

    obj_link.set_joint_properties(
        joint_type="revolute",
        limits=[[-np.pi, np.pi]],
        pose_in_child=sapien.Pose(
            [bbox_points[0], bbox_points[1], bbox_points[2]], [1, 0, 0, 0]),
        pose_in_parent=sapien.Pose(
            [bbox_points[0], bbox_points[1], bbox_points[2]], [1, 0, 0, 0]))

    obj_link.set_parent(slider_link.get_index())
    collision = builder.build(fix_root_link=True)

    collision.get_active_joints()[-1].set_drive_property(200,
                                                         60,
                                                         10,
                                                         mode="force")
    collision.set_qpos([np.pi / 2])
    collision.set_drive_target([np.pi / 2])

    return collision, height
