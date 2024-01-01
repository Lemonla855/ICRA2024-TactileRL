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
import pymeshlab
from PIL import Image
import pdb
import trimesh
from scipy.spatial.transform import Rotation as R
import json
from pathlib import Path

from hand_teleop.utils.shapenet_utils import get_shapenet_root_dir, SHAPENET_CAT, load_shapenet_object_list
from hand_teleop.utils.egad_object_utils import get_egad_root_dir, load_egad_scale, EGAD_ORIENTATION, EGAD_RESCALE
from hand_teleop.utils.modelnet_object_utils import get_modelnet_root_dir, MODELNET40_SCALE, MODELNET40_ANYTRAIN
from hand_teleop.utils.ycb_object_utils import YCB_CLASSES, INVERSE_YCB_CLASSES, get_ycb_root_dir, YCB_OBJECT_NAMES, YCB_OBJECT_NAMES_EXIT
from hand_teleop.utils.partnet_class_utils import KINIFE_ANYTRAIN, LIGHTER_ANYTRAIN, DISPENSER_ANYTRAIN, bottle_json, knife_json, dispenser_json, lighter_json
from hand_teleop.utils.facucet_setting import FAUCET_SINGLE
from hand_teleop.utils.partnet_utils import BOTTLE_ANYTRAIN

from hand_teleop.utils.bottle4_helper import CYLINDER_ANYTRAIN, CAPSULE_ANYTRAIN, THINCAPSULE2_ANYTRAIN, THINCAPSULE3_ANYTRAIN, CONE_ANYTRAIN, TORUS_ANYTRAIN, UPTORUS_ANYTRAIN, THINCAPSULE_ANYTRAIN, ANY_TRAIN


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
        scale = info["scales"]
        # height = info["height"]

        mesh = trimesh.load(visual_file)
        vertices = mesh.vertices

        scale_vertices = vertices * np.array(scale * 3).reshape(3)

        #rotate mesh according to the local pose
        quat = [0.7071, 0, 0, 0.7071]
        r = R.from_quat(quat)
        rotation = r.as_matrix()
        rotated_vertices = (rotation @ scale_vertices.T).T

        obj_meshes.append([rotated_vertices, mesh.faces])

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
#         if split == "eval":
#             scale = EGAD_SCALE[split][model_id]["scales"]
#             scales = np.array(scale * 3)
#             scales[2] *= 2
#         else:
#             raise NotImplementedError
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

    egad_meshes = []
    for model_id in names:
        split = "train" if "_" in model_id else "eval"
        if split == "eval":
            scale = EGAD_SCALE[split][model_id]["scales"]
            scales = np.array(scale * 3)
            scales[2] *= 2
        else:
            raise NotImplementedError
        visual_file = str(egad_dir / "egad_{split}_set" /
                          f"{model_id}.obj").format(split=split)
        if model_id in EGAD_RESCALE.keys():
            scales = EGAD_RESCALE[model_id]

        mesh = trimesh.load(visual_file)
        vertices = mesh.vertices

        scale_vertices = vertices * np.array(scales).reshape(3)

        #rotate mesh according to the initial pose

        quat = np.array(EGAD_ORIENTATION[model_id])[[
            1, 2, 3, 0
        ]]  #scipy is xyzw, EGAD_ORIENTATION is wxyz
        r = R.from_quat(quat)
        rotation = r.as_matrix()
        rotated_vertices = (rotation @ scale_vertices.T).T

        egad_meshes.append([[[rotated_vertices]], mesh.faces])

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
#   subdivide the mesh
#=======================================================
def add_vertices(ms):
    ms.meshing_remove_duplicate_faces()
    ms.meshing_repair_non_manifold_edges()
    ms.meshing_repair_non_manifold_vertices()
    ms.meshing_surface_subdivision_midpoint(iterations=2)

    return ms


#=======================================================
#   save different mesh
#=======================================================


def save_visual_mesh(
    object,
    object_category,
    object_name,
    collision=True,
):

    meshes = []
    links_name = []

    import os

    if collision:
        mesh_type = "collision"
    else:
        mesh_type = "visual"

    if not os.path.exists("./assets/partnet-mobility-dataset/" +
                          object_category + "/" + mesh_type):
        os.mkdir("./assets/partnet-mobility-dataset/" + object_category + "/" +
                 mesh_type)

    if not os.path.exists("./assets/partnet-mobility-dataset/" +
                          object_category + "/" + mesh_type + "/" +
                          object_name):
        os.mkdir("./assets/partnet-mobility-dataset/" + object_category + "/" +
                 mesh_type + "/" + object_name)

    for link in object.get_links():

        link_name = link.get_name()

        link_mesh = []

        #load visual body for the link

        if collision:
            visual_bodies = link.get_collision_visual_bodies()

        else:
            visual_bodies = link.get_visual_bodies()
        collision_shape = link.get_collision_shapes()
        # for c in collision_shape:
        #     print(c.contact_offset)

        for body_index, body in enumerate(visual_bodies):

            body_shape = body.get_render_shapes()

            # pdb.set_trace()
            # collision_shape = link.get_collision_shapes()

            #pose
            body_pose = body.local_pose  #wxyz
            quat = body_pose.q[[1, 2, 3, 0]]  #xyzw
            pos = body_pose.p

            # print(body.scale)
            scales = body.scale

            for index, shape in enumerate(body_shape):

                shape_mesh = shape.mesh
                vertices = shape_mesh.vertices

                # if body.type == "box":  # for box, need change size
                #     half_lengths = body.half_lengths
                #     vertices = vertices * half_lengths.reshape(3)

                #rotate mesh according to the local pose
                # scale_vertices = vertices * np.array(scales).reshape(3)
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
            links_name.append(str(link_name))
            link_trimesh.export("./assets/partnet-mobility-dataset/" +
                                object_category + "/" + mesh_type + "/" +
                                object_name + "/" + link_name + ".stl")

            if collision:

                if not os.path.exists("./assets/partnet-mobility-dataset/" +
                                      object_category + "/" + "pymeshlab" +
                                      "/"):
                    os.mkdir("./assets/partnet-mobility-dataset/" +
                             object_category + "/" + "pymeshlab" + "/")

                if not os.path.exists("./assets/partnet-mobility-dataset/" +
                                      object_category + "/" + "pymeshlab" +
                                      "/" + object_name):
                    os.mkdir("./assets/partnet-mobility-dataset/" +
                             object_category + "/" + "pymeshlab" + "/" +
                             object_name)

                ms = pymeshlab.MeshSet()
                ms.load_new_mesh("./assets/partnet-mobility-dataset/" +
                                 object_category + "/" + mesh_type + "/" +
                                 object_name + "/" + link_name + ".stl")
                new_ms = add_vertices(ms)
                new_ms.save_current_mesh("./assets/partnet-mobility-dataset/" +
                                         object_category + "/" + "pymeshlab" +
                                         "/" + object_name + "/" + link_name +
                                         ".stl")

    # export mesh
    partnet_meshes = trimesh.util.concatenate(meshes)

    partnet_meshes.export("./assets/partnet-mobility-dataset/" +
                          object_category + "/" + mesh_type + "/" +
                          object_name + "/" + object_name + ".stl")

    if collision:

        np.savetxt("./assets/partnet-mobility-dataset/" + object_category +
                   "/" + mesh_type + "/" + object_name + '/link.txt',
                   links_name,
                   delimiter=" ",
                   fmt="%s")
        np.savetxt("./assets/partnet-mobility-dataset/" + object_category +
                   "/" + "pymeshlab" + "/" + object_name + '/link.txt',
                   links_name,
                   delimiter=" ",
                   fmt="%s")


#=======================================================
#   save partnet mesh
#=======================================================
def save_partnet_mesh(object, object_category, object_name):

    save_visual_mesh(
        object,
        object_category,
        object_name,
        collision=True,
    )
    save_visual_mesh(
        object,
        object_category,
        object_name,
        collision=False,
    )

    # meshes = []
    # import os

    # if not os.path.exists("./assets/partnet-mobility-dataset/" +
    #                       object_category + "/render_mesh"):
    #     os.mkdir("./assets/partnet-mobility-dataset/" + object_category +
    #              "/render_mesh")

    # if not os.path.exists("./assets/partnet-mobility-dataset/" +
    #                       object_category + "/render_mesh/" + object_name):
    #     os.mkdir("./assets/partnet-mobility-dataset/" + object_category +
    #              "/render_mesh/" + object_name)

    # links_name = []

    # for link in object.get_links():

    #     link_name = link.get_name()

    #     link_mesh = []

    #     #load visual body for the link
    #     visual_bodies = link.get_collision_visual_bodies()

    #     for body_index, body in enumerate(visual_bodies):

    #         body_shape = body.get_render_shapes()

    #         #pose
    #         body_pose = body.local_pose  #wxyz
    #         quat = body_pose.q[[1, 2, 3, 0]]  #xyzw
    #         pos = body_pose.p

    #         for index, shape in enumerate(body_shape):

    #             shape_mesh = shape.mesh
    #             vertices = shape_mesh.vertices

    #             # if body.type == "box":  # for box, need change size
    #             #     half_lengths = body.half_lengths
    #             #     vertices = vertices * half_lengths.reshape(3)

    #             #rotate mesh according to the local pose
    #             r = R.from_quat(quat)
    #             rotation = r.as_matrix()
    #             vertices = (rotation @ vertices.T).T + pos

    #             body_trimesh = trimesh.Trimesh(
    #                 vertices=vertices,
    #                 faces=shape_mesh.indices.reshape(-1, 3),
    #             )

    #             link_mesh.append(body_trimesh)

    #             meshes.append(body_trimesh)
    #             # if link_name != "frame":
    #             #     door_exclude_frames_meshes.append(body_trimesh)
    #     if bool(link_mesh):
    #         link_trimesh = trimesh.Scene(link_mesh)
    #         links_name.append(link_name)
    #         link_trimesh.export("./assets/partnet-mobility-dataset/" +
    #                             object_category + "/render_mesh/" +
    #                             object_name + "/" + link_name + ".stl")

    # # export mesh
    # partnet_meshes = trimesh.util.concatenate(meshes)

    # partnet_meshes.export(
    #     str("./assets/partnet-mobility-dataset/" + object_category +
    #         "/render_mesh/") + "/" + object_name + ".stl")

    # np.savetxt("./assets/partnet-mobility-dataset/" + object_category +
    #            "/render_mesh/" + object_name + '/link.txt',
    #            links_name,
    #            delimiter=" ",
    #            fmt="%s")


def load_partnet_object(object_category, model_names):

    partnet_mesh = []
    import os

    if "any" in model_names[0]:

        if object_category in ["bottle"]:

            model_names = BOTTLE_ANYTRAIN

        if object_category in ["bottle4"]:
            if "cylinder" in model_names[0]:
                model_names = CYLINDER_ANYTRAIN

            elif "train" in model_names[0]:
                model_names = ANY_TRAIN

            elif "any_capsule" in model_names[0]:
                model_names = CAPSULE_ANYTRAIN

            elif "thincapsule" in model_names[0]:
                model_names = THINCAPSULE_ANYTRAIN

            elif "thincapsule2" in model_names[0]:
                model_names = THINCAPSULE2_ANYTRAIN

            elif "thincapsule3" in model_names[0]:
                model_names = THINCAPSULE3_ANYTRAIN

            elif "cone" in model_names[0]:
                model_names = CONE_ANYTRAIN

            elif "torus" in model_names[0]:
                model_names = TORUS_ANYTRAIN

            elif "uptorus" in model_names[0]:
                model_names = UPTORUS_ANYTRAIN

        print("The rendering list is", model_names)

        if object_category == "knife":
            model_names = KINIFE_ANYTRAIN

        if object_category == "dispenser":
            model_names = DISPENSER_ANYTRAIN

        if object_category == "lighter":
            model_names = LIGHTER_ANYTRAIN

    for name in model_names:

        meshes_vertices = []
        meshes_faces = []
        face_count = 0

        if name in FAUCET_SINGLE:
            link_names = [
                str(
                    np.loadtxt("./assets/partnet-mobility-dataset/" +
                               object_category + "/pymeshlab/" + name +
                               "/link.txt",
                               dtype=str))
            ]
        else:

            link_names = np.loadtxt("./assets/partnet-mobility-dataset/" +
                                    object_category + "/pymeshlab/" + name +
                                    "/link.txt",
                                    dtype=str)[-1:]

        for link_name in link_names:

            mesh = trimesh.load("./assets/partnet-mobility-dataset/" +
                                object_category + "/pymeshlab/" + name + "/" +
                                link_name + ".stl")
            vertices = mesh.vertices

            scales = np.array([1] * 3)

            scale_vertices = vertices * np.array(scales).reshape(3)

            meshes_vertices.append(scale_vertices)

            face = mesh.faces + face_count
            face_count += len(vertices)

            meshes_faces.append(face)
            # partnet_mesh.append([scale_vertices, mesh.faces])

        # partnet_mesh.append(meshes_vertices)
        # partnet_mesh.append(np.concatenate(meshes_faces, axis=0))

        partnet_mesh.append(
            [meshes_vertices,
             np.concatenate(meshes_faces, axis=0)])

    return partnet_mesh, model_names
