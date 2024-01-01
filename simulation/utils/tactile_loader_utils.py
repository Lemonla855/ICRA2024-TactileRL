from logging import captureWarnings
from statistics import mode

from typing import List, Union
from urdfpy import URDF
from scipy.spatial.transform import Rotation as R
import sapien.core as sapien
import torch
import numpy as np
import cv2
import imageio
import pdb

from PIL import Image
import pdb
import trimesh
from scipy.spatial.transform import Rotation as R
import json
from pathlib import Path

from hand_teleop.utils.shapenet_utils import get_shapenet_root_dir, SHAPENET_CAT, load_shapenet_object_list
from hand_teleop.utils.egad_object_utils import get_egad_root_dir, load_egad_scale, EGAD_ORIENTATION, EGAD_RESCALE, EGAD_LIST
from hand_teleop.utils.modelnet_object_utils import get_modelnet_root_dir, MODELNET40_SCALE, MODELNET40_ANYTRAIN
from hand_teleop.utils.ycb_object_utils import YCB_CLASSES, INVERSE_YCB_CLASSES, get_ycb_root_dir, YCB_OBJECT_NAMES, YCB_OBJECT_NAMES_EXIT
from hand_teleop.utils.partnet_utils import PARTNET_HEIGHT, BOTTLE_ANYTRAIN, BREAKING_ANYTRAIN, BOX_ANYTRAIN, BUCKET_ANYTRAIN, BREAKING_BOTTLE_ANYTRAIN, PARTNET_ORIENTATION, PARTNET_SCALE, SPOON_ANYTRAIN, KITCHEN_ANYTRAIN, KITCHEN_ORIENTATION, SPOON_ORIENTATION, PEN_ANYTRAIN2, USB_ANYTRAIN, KNIFE_ANYTRAIN, PEN_ANYTRAIN_NOVEL, USB_ANYTRAIN_NOVEL, SPOON_NOVEL_ANYTRAIN


#=======================================================
#   load ycb object for rendering
#=======================================================
def load_ycb_object(
    object_name,
    scale=1,
):
    YCB_ROOT = get_ycb_root_dir()
    obj_meshes = []

    if object_name == "any_train" or object_name == "any_eval":
        for ycb_name in YCB_OBJECT_NAMES_EXIT:

            visual_file = str(YCB_ROOT / "visual" / ycb_name /
                              "textured_simple.obj")

            ycb_mesh = trimesh.load(visual_file)

            vertices = ycb_mesh.vertices
            scale_vertices = vertices * np.array([scale] * 3).reshape(3)

            obj_meshes.append([scale_vertices, ycb_mesh.faces])

        return obj_meshes, YCB_OBJECT_NAMES_EXIT

    # for object_name in object_names:
    ycb_id = INVERSE_YCB_CLASSES[object_name]
    ycb_name = YCB_CLASSES[ycb_id]
    visual_file = str(YCB_ROOT / "visual" / ycb_name / "textured_simple.obj")
    ycb_mesh = trimesh.load(visual_file)

    vertices = ycb_mesh.vertices
    scale_vertices = vertices * np.array([scale] * 3).reshape(3)

    obj_meshes.append([scale_vertices, ycb_mesh.faces])

    return obj_meshes, [object_name]


# def load_shapenet_object_list():
#     cat_dict = {}
#     info_path = get_shapenet_root_dir() / "info.json"
#     with info_path.open("r") as f:
#         cat_scale = json.load(f)
#     for cat in SHAPENET_CAT:
#         object_list_file = get_shapenet_root_dir() / f"{cat}.txt"
#         with object_list_file.open("r") as f:
#             cat_object_list = f.read().split("\n")
#         cat_dict[cat] = {}
#         for model_id in cat_object_list:
#             cat_dict[cat][model_id] = cat_scale[cat][model_id]
#         cat_dict[cat]["train"] = cat_object_list[:10]
#         cat_dict[cat]["eval"] = cat_object_list[10:]

#     return cat_dict

#=======================================================
#   load object from shapeNet
#=======================================================

CAT_DICT = load_shapenet_object_list()


def load_shapenet_object(cat_id: str, object_name=None, key: str = "train"):

    shapenet_dir = get_shapenet_root_dir()

    object_list = CAT_DICT[cat_id][key]

    obj_meshes = []

    if object_name is not None:
        object_list = [object_name]

    for model_id in object_list:
        # collision_file = str(shapenet_dir / cat_id / model_id / "convex.obj")
        visual_file = str(shapenet_dir / cat_id / model_id / "convex.obj")

        info = CAT_DICT[cat_id][model_id]
        scale = [1]
        # height = info["height"]

        mesh = trimesh.load(visual_file)
        vertices = mesh.vertices

        scale_vertices = vertices * np.array(scale * 3).reshape(3)

        #rotate mesh according to the local pose
        quat = [0.7071, 0, 0, 0.7071]
        r = R.from_quat(quat)
        rotation = r.as_matrix()
        rotated_vertices = (rotation @ scale_vertices.T).T

        obj_meshes.append([[rotated_vertices], mesh.faces])

    return obj_meshes, object_list


# #=======================================================
# #   load object from egad
# #=======================================================
# #TODO:verify the function of loading egad objects
# def load_egad_object(names):
#     # Source: https://github.com/haosulab/ManiSkill2022/tree/main/scripts/jigu/egad
#     egad_dir = get_egad_root_dir()
#     EGAD_SCALE = load_egad_scale()

#     #TODO:verify the scale for egad train objects is 1
#     scale = 1

#     egad_meshes = []
#     for model_id in names:
#         split = "train" if "_" in model_id else "eval"
#         # if split == "eval":
#         #     scale = EGAD_SCALE[split][model_id]["scales"]
#         #     scales = np.array(scale * 3)
#         #     scales[2] *= 2
#         # else:
#         #     raise NotImplementedError
#         visual_file = str(egad_dir / "egad_{split}_set" /
#                           f"{model_id}.obj").format(split=split)
#         if model_id in EGAD_RESCALE.keys():
#             scales = EGAD_RESCALE[model_id]

#         mesh = trimesh.load(visual_file)
#         vertices = mesh.vertices

#         scale_vertices = vertices * np.array(scales).reshape(3)

#         #rotate mesh according to the initial pose

#         quat = np.array(EGAD_ORIENTATION[model_id])[[
#             1, 2, 3, 0
#         ]]  #scipy is xyzw, EGAD_ORIENTATION is wxyz
#         r = R.from_quat(quat)
#         rotation = r.as_matrix()
#         rotated_vertices = (rotation @ scale_vertices.T).T

#         egad_meshes.append([rotated_vertices, mesh.faces])

#     return egad_meshes, names

#=======================================================
#   load object from modelnet
#=======================================================


def load_egad_object(names):
    # Source: https://github.com/haosulab/ManiSkill2022/tree/main/scripts/jigu/egad
    egad_dir = get_egad_root_dir()
    EGAD_SCALE = load_egad_scale()

    #TODO:verify the scale for egad train objects is 1
    scale = 1

    # if "front" in names:
    names = EGAD_LIST["front"]
    # pdb.set_trace()

    egad_meshes = []
    for model_id in names:
        split = "train" if "_" in model_id else "eval"
        # scale = EGAD_SCALE[split][model_id]["scales"]
        # scales = np.array(scale * 3)
        scale = [1]
        scales = np.array(scale * 3)
        # if split == "eval":
        #     scale = EGAD_SCALE[split][model_id]["scales"]
        #     scales = np.array(scale * 3)
        #     scales[2] *= 2
        # else:
        #     raise NotImplementedError
        visual_file = str(egad_dir / "egad_{split}_set" /
                          f"{model_id}.obj").format(split=split)
        # if model_id in EGAD_RESCALE.keys():
        #     scales = EGAD_RESCALE[model_id]

        mesh = trimesh.load(visual_file)
        vertices = mesh.vertices

        scale_vertices = vertices * np.array(scales).reshape(3)

        #rotate mesh according to the initial pose

        # quat = np.array(EGAD_ORIENTATION[model_id])[[
        #     1, 2, 3, 0
        # ]]  #scipy is xyzw, EGAD_ORIENTATION is wxyz
        quat = [1, 0, 0, 0]
        r = R.from_quat(quat)
        rotation = r.as_matrix()
        rotated_vertices = (rotation @ scale_vertices.T).T

        egad_meshes.append([[rotated_vertices], mesh.faces])
        # import matplotlib.pyplot as plt
        # ax = plt.axes(projection='3d')
        # # points_pixel = pixel_coords[0].cpu().view(-1, 1, 3)
        # # ax.scatter(points_pixel[:, :, 0].reshape(-1),
        # #            points_pixel[:, :, 1].reshape(-1),
        # #            points_pixel[:, :, 2].reshape(-1),
        # #            c=points_pixel[:, :, 0].reshape(-1))
        # points_pixel = rotated_vertices.reshape(-1, 3)
        # ax.scatter(points_pixel[:, 0].reshape(-1),
        #            points_pixel[:, 1].reshape(-1),
        #            points_pixel[:, 2].reshape(-1),
        #            c='r')

    return egad_meshes, names


def load_modelnet_object(model_name):
    modelnet_dir, scale = get_modelnet_root_dir()

    if model_name == "any_train":
        model_names = MODELNET40_ANYTRAIN
    else:
        model_names = [model_name]

    modelnet_mesh = []

    for name in model_names:
        collision_name = str(name.split("_")[0]) + "_vhac"

        visual_file = str(modelnet_dir / collision_name / f"{name}.obj")
        mesh = trimesh.load(visual_file)

        vertices = mesh.vertices
        if name in MODELNET40_SCALE.keys():
            scales = np.array([scale[name.split("_")[0]][name]["scale"]] * 3)
        else:
            raise NotImplementedError

        scale_vertices = vertices * np.array(scales).reshape(3)
        modelnet_mesh.append([scale_vertices, mesh.faces])

    return modelnet_mesh, model_names


import os
import pymeshlab


def load_partnet_actor_object(object_cat, model_name, novel=False):

    if object_cat == "pen" and model_name == "any_train":
        if novel:
            model_names = PEN_ANYTRAIN_NOVEL
        else:
            model_names = PEN_ANYTRAIN2
    elif object_cat == "USB" and model_name == "any_train":
        if novel:
            model_names = USB_ANYTRAIN_NOVEL
        else:
            model_names = USB_ANYTRAIN
    elif object_cat == "knife" and model_name == "any_train":
        model_names = KNIFE_ANYTRAIN
    elif object_cat == "bottle" and model_name == "any_train":
        model_names = BOTTLE_ANYTRAIN
    elif object_cat == "breaking" and model_name == "any_train":
        model_names = BREAKING_ANYTRAIN
    elif object_cat == "breaking_bottle" and model_name == "any_train":
        model_names = BREAKING_BOTTLE_ANYTRAIN
    elif object_cat == "box" and model_name == "any_train":
        model_names = BOX_ANYTRAIN
    elif object_cat == "bucket" and model_name == "any_train":
        model_names = BUCKET_ANYTRAIN
    else:
        model_names = [model_name]

    spoon_mesh = []

    for model_name in model_names:

        # if model_name not in KITCHEN_ORIENTATION.keys():
        #     r = R.from_euler('xyz',
        #                      SPOON_ORIENTATION[model_name],
        #                      degrees=True)
        # else:
        #     r = R.from_euler('xyz',
        #                      KITCHEN_ORIENTATION[model_name],
        #                      degrees=True)
        # quat = r.as_quat()
        multiple_vertices = []
        multiple_faces = []

        vertices_count = 0

        # for file in collision_files:

        mesh = trimesh.load("./assets/partnet-mobility-dataset/collisions/" +
                            model_name + ".stl")

        vertices = mesh.vertices

        scale = [1]

        scale_vertices = vertices * np.array(scale * 3).reshape(3)

        # r = R.from_quat(quat)
        # rotation = r.as_matrix()

        # rotated_vertices = (rotation @ scale_vertices.T).T

        multiple_vertices.append(vertices)
        multiple_faces.append(mesh.faces + vertices_count)
        vertices_count += len(vertices)

        multiple_vertices = np.concatenate(multiple_vertices, axis=0)
        multiple_faces = np.concatenate(multiple_faces, axis=0)

        spoon_mesh.append([[multiple_vertices], mesh.faces])

    return spoon_mesh, model_names


def load_spoon_object(object_cat, model_name):

    if object_cat == "kitchen" and model_name == "any_train":
        model_names = KITCHEN_ANYTRAIN
    elif object_cat == "spoon" and model_name == "any_train":
        model_names = SPOON_ANYTRAIN
    else:
        model_names = [model_name]

    spoon_mesh = []

    for model_name in model_names:

        collision_files = os.listdir("./assets/spoon/visual_center/" +
                                     model_name)

        # if model_name not in KITCHEN_ORIENTATION.keys():
        #     r = R.from_euler('xyz',
        #                      SPOON_ORIENTATION[model_name],
        #                      degrees=True)
        # else:
        #     r = R.from_euler('xyz',
        #                      KITCHEN_ORIENTATION[model_name],
        #                      degrees=True)
        # quat = r.as_quat()
        multiple_vertices = []
        multiple_faces = []

        vertices_count = 0

        # for file in collision_files:

        mesh = trimesh.load("./assets/spoon/visual_center/" + model_name +
                            "/" + model_name + ".stl")

        vertices = mesh.vertices

        scale = [1]

        scale_vertices = vertices * np.array(scale * 3).reshape(3)

        # r = R.from_quat(quat)
        # rotation = r.as_matrix()

        # rotated_vertices = (rotation @ scale_vertices.T).T

        multiple_vertices.append(vertices)
        multiple_faces.append(mesh.faces + vertices_count)
        vertices_count += len(vertices)

        multiple_vertices = np.concatenate(multiple_vertices, axis=0)
        multiple_faces = np.concatenate(multiple_faces, axis=0)

        spoon_mesh.append([[multiple_vertices], mesh.faces])

    return spoon_mesh, model_names


#=======================================================
#   load door mesh path
#=======================================================


def get_door_root_dir():
    current_dir = Path(__file__).parent
    door_dir = current_dir.parent.parent / "assets" / "door" / "mesh"
    return door_dir


#=======================================================
#   save door mesh
#=======================================================
def save_door_mesh(table_door):
    door_mesh_path = get_door_root_dir()

    door_meshes = []
    door_exclude_frames_meshes = []

    for link in table_door.get_links():

        link_name = link.get_name()
        #load visual body for the link
        visual_bodies = link.get_visual_bodies()

        for body_index, body in enumerate(visual_bodies):

            body_shape = body.get_render_shapes()

            #pose
            body_pose = body.local_pose  #wxyz
            quat = body_pose.q[[1, 2, 3, 0]]  #xyzw
            pos = body_pose.p

            for index, shape in enumerate(body_shape):

                shape_mesh = shape.mesh
                vertices = shape_mesh.vertices

                if body.type == "box":  # for box, need change size
                    half_lengths = body.half_lengths
                    vertices = vertices * half_lengths.reshape(3)

                #rotate mesh according to the local pose
                r = R.from_quat(quat)
                rotation = r.as_matrix()
                vertices = (rotation @ vertices.T).T + pos

                body_trimesh = trimesh.Trimesh(
                    vertices=vertices,
                    faces=shape_mesh.indices.reshape(-1, 3),
                )
                body_trimesh.export(
                    str(door_mesh_path) + "/" +
                    str(link_name + "_body" + str(body_index) + "_" +
                        str(index)) + ".stl")

                door_meshes.append(body_trimesh)
                if link_name != "frame":
                    door_exclude_frames_meshes.append(body_trimesh)

    # export mesh
    door = trimesh.util.concatenate(door_meshes)
    door.export(str(door_mesh_path) + "/"
                "door.stl")
    door_exclude_frames = trimesh.util.concatenate(door_exclude_frames_meshes)
    door_exclude_frames.export(
        str(door_mesh_path) + "/"
        "door_exclude_frames.stl")


def get_partnet_root_dir():
    current_dir = Path(__file__).parent
    door_dir = current_dir.parent.parent / "assets" / "partnet-mobility-dataset" / "render_mesh"
    return door_dir


#=======================================================
#   save partnet actor mesh
#=======================================================
def save_partnet_actor_mesh(object, object_name):
    partnetr_mesh_path = get_partnet_root_dir()

    meshes = []
    link_mesh = []
    import os

    result_path = Path("./assets/partnet-mobility-dataset/render_mesh/" +
                       object_name)
    result_path.mkdir(exist_ok=True, parents=True)

    visual_bodies = object.get_collision_shapes()

    for body_index, body in enumerate(visual_bodies):

        body_pose = body.get_local_pose()  #wxyz
        quat = body_pose.q[[1, 2, 3, 0]]  #xyzw
        pos = body_pose.p
        r = R.from_quat(quat)
        rotation = r.as_matrix()

        vertices = body.geometry.vertices

        vertices = (rotation @ vertices.T).T + pos

        body_trimesh = trimesh.Trimesh(
            vertices=vertices,
            faces=body.geometry.indices.reshape(-1, 3),
        )
        link_mesh.append(body_trimesh)
        meshes.append(body_trimesh)
        # if link_name != "frame":
        #     door_exclude_frames_meshes.append(body_trimesh)
    if not os.path.exists("./assets/partnet-mobility-dataset/collisions/"):
        os.mkdir("./assets/partnet-mobility-dataset/collisions/")

    if bool(link_mesh):
        link_trimesh = trimesh.Scene(link_mesh)

        link_trimesh.export("./assets/partnet-mobility-dataset/collisions/" +
                            "/" + object_name + ".stl")


#=======================================================
#   save spoon mesh
#=======================================================
def save_spoon_mesh(object, object_name):
    partnetr_mesh_path = get_partnet_root_dir()

    meshes = []
    import os

    result_path = Path("./assets/spoon/visual_center/" + object_name)
    result_path.mkdir(exist_ok=True, parents=True)

    links_name = []

    link_name = object.get_name()

    link_mesh = []

    visual_bodies = object.get_collision_shapes()

    for body_index, body in enumerate(visual_bodies):

        body_pose = body.get_local_pose()  #wxyz
        quat = body_pose.q[[1, 2, 3, 0]]  #xyzw
        pos = body_pose.p
        r = R.from_quat(quat)
        rotation = r.as_matrix()

        vertices = body.geometry.vertices

        vertices = (rotation @ vertices.T).T + pos

        body_trimesh = trimesh.Trimesh(
            vertices=vertices,
            faces=body.geometry.indices.reshape(-1, 3),
        )
        link_mesh.append(body_trimesh)
        meshes.append(body_trimesh)
        if bool(link_mesh):
            link_trimesh = trimesh.Scene(link_mesh)
            links_name.append(link_name)
            link_trimesh.export("./assets/spoon/visual_center/" + object_name +
                                "/" + object_name + ".stl")

    # export mesh
    # partnet_meshes = trimesh.util.concatenate(meshes)
    # pdb.set_trace()
    # partnet_meshes.export(str(partnetr_mesh_path) + "/" + object_name + ".stl")
    # print(links_name)
    # np.savetxt("./assets/partnet-mobility-dataset/render_mesh/" + object_name +
    #            '/link.txt',
    #            links_name,
    #            delimiter=" ",
    #            fmt="%s")


#=======================================================
#   save partnet mesh
#=======================================================
def save_partnet_mesh(object, object_name):
    partnetr_mesh_path = get_partnet_root_dir()

    meshes = []
    import os

    if not os.path.exists("./assets/partnet-mobility-dataset/render_mesh/" +
                          object_name):
        os.mkdir("./assets/partnet-mobility-dataset/render_mesh/" +
                 object_name)

    links_name = []

    for link in object.get_links():

        link_name = link.get_name()

        link_mesh = []

        #load visual body for the link
        visual_bodies = link.get_visual_bodies()

        for body_index, body in enumerate(visual_bodies):

            body_shape = body.get_render_shapes()

            #pose
            body_pose = body.local_pose  #wxyz
            quat = body_pose.q[[1, 2, 3, 0]]  #xyzw
            pos = body_pose.p

            for index, shape in enumerate(body_shape):

                shape_mesh = shape.mesh
                vertices = shape_mesh.vertices

                # if body.type == "box":  # for box, need change size
                #     half_lengths = body.half_lengths
                #     vertices = vertices * half_lengths.reshape(3)

                #rotate mesh according to the local pose
                r = R.from_quat(quat)
                rotation = r.as_matrix()
                vertices = (rotation @ vertices.T).T + pos

                body_trimesh = trimesh.Trimesh(
                    vertices=vertices,
                    faces=shape_mesh.indices.reshape(-1, 3),
                )

                link_mesh.append(body_trimesh)

                meshes.append(body_trimesh)
                # if link_name != "frame":
                #     door_exclude_frames_meshes.append(body_trimesh)
        if bool(link_mesh):
            link_trimesh = trimesh.Scene(link_mesh)
            links_name.append(link_name)
            link_trimesh.export(
                "./assets/partnet-mobility-dataset/render_mesh/" +
                object_name + "/" + link_name + ".stl")

    # export mesh
    # partnet_meshes = trimesh.util.concatenate(meshes)
    # pdb.set_trace()
    # partnet_meshes.export(str(partnetr_mesh_path) + "/" + object_name + ".stl")
    print(links_name)
    np.savetxt("./assets/partnet-mobility-dataset/render_mesh/" + object_name +
               '/link.txt',
               links_name,
               delimiter=" ",
               fmt="%s")


def load_partnet_object(model_names):

    partnet_mesh = []
    import os

    for name in model_names:

        meshes_vertices = []
        meshes_faces = []
        face_count = 0

        link_names = np.loadtxt(
            "./assets/partnet-mobility-dataset/render_mesh/" + name +
            "/link.txt",
            dtype=str)

        for link_name in link_names:
            mesh = trimesh.load(
                "./assets/partnet-mobility-dataset/render_mesh/" + name + "/" +
                link_name + ".stl")
            vertices = mesh.vertices
            scales = np.array([PARTNET_SCALE[name]] * 3)

            scale_vertices = vertices * np.array(scales).reshape(3)

            meshes_vertices.append(scale_vertices)

            face = mesh.faces + face_count
            face_count += len(vertices)

            meshes_faces.append(face)
            # partnet_mesh.append([scale_vertices, mesh.faces])

        partnet_mesh.append(meshes_vertices)
        partnet_mesh.append(np.concatenate(meshes_faces, axis=0))

    return partnet_mesh, model_names
