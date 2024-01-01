import numpy as np
import torch
import sapien.core as sapien
from scipy.spatial.transform import Rotation as R
from pathlib import Path
import trimesh
import os
import pdb


def compute_smooth_shading_normal_np(vertices, indices):
    """
    Compute the vertex normal from vertices and triangles with numpy
    Args:
        vertices: (n, 3) to represent vertices position
        indices: (m, 3) to represent the triangles, should be in counter-clockwise order to compute normal outwards
    Returns:
        (n, 3) vertex normal

    References:
        https://www.iquilezles.org/www/articles/normals/normals.htm
    """
    v1 = vertices[indices[:, 0]]
    v2 = vertices[indices[:, 1]]
    v3 = vertices[indices[:, 2]]
    face_normal = np.cross(v2 - v1,
                           v3 - v1)  # (n, 3) normal without normalization to 1

    vertex_normal = np.zeros_like(vertices)
    vertex_normal[indices[:, 0]] += face_normal
    vertex_normal[indices[:, 1]] += face_normal
    vertex_normal[indices[:, 2]] += face_normal
    vertex_normal /= np.linalg.norm(vertex_normal, axis=1, keepdims=True)
    return vertex_normal


def compute_batch_smooth_shading_normal_torch(vertices, indices):
    """
    Compute the vertex normal from vertices and triangles with torch
    Args:
        vertices: (b, n, 3) to represent vertices position
        indices: (b, m, 3) to represent the triangles
    Returns:
        (b, n, 3) vertex normal
    """
    pass


def save_articulated_mesh(object, save_dir, collision=True):

    save_path = Path(save_dir)
    save_path.mkdir(exist_ok=True, parents=True)

    meshes = []
    body_meshes = []

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

            for index, shape in enumerate(body_shape):

                shape_mesh = shape.mesh
                vertices = shape_mesh.vertices

                # if body.type == "box":  # for box, need change size
                #     half_lengths = body.half_lengths
                #     vertices = vertices * half_lengths.reshape(3)

                #rotate mesh according to the local pose
                # print(vertices.shape,body.scale)
                # vertices = vertices * np.array(body.scale).reshape(3)
                r = R.from_quat(quat)
                rotation = r.as_matrix()

                vertices = (rotation @ vertices.T).T + pos

                body_trimesh = trimesh.Trimesh(
                    vertices=vertices,
                    faces=shape_mesh.indices.reshape(-1, 3),
                )

                link_mesh.append(body_trimesh)

                meshes.append(body_trimesh)

        if bool(link_mesh):
            body_meshes.append(link_mesh)
            link_trimesh = trimesh.Scene(link_mesh)

            if collision:

                save_collision_path = Path(save_dir) / "collision"
                save_collision_path.mkdir(exist_ok=True, parents=True)

                link_trimesh.export(
                    str(save_collision_path) + "/" + str(link_name + ".stl"))

    body_trimesh = trimesh.Scene(body_meshes)
    body_trimesh.export(str(save_collision_path) + "/" + str("body.stl"))
