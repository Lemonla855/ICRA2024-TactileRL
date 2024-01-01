#Important installation: # install pycuda first, https://github.com/inducer/pycuda and

# from tkinter import E
from typing import List
import torch
import numpy as np
import cv2
from typing import List
import matplotlib.pyplot as plt
# import matplotlib.pyplot as plt
import time
from typing import Union
# datastructures
from pytorch3d.structures import Meshes

# 3D transformations functions
from pytorch3d.transforms import quaternion_to_matrix, Transform3d, quaternion_invert, matrix_to_quaternion, quaternion_multiply

# rendering components
from pytorch3d.renderer import (RasterizationSettings, MeshRasterizer,
                                OpenGLPerspectiveCameras)

from pytorch3d.structures.meshes import (
    join_meshes_as_batch,
    join_meshes_as_scene,
    Meshes,
)

#=====================================faster ===========================
# import os
# os.environ["PYOPENGL_PLATFORM"] = "egl"
# import OpenGL.EGL
# from pytorch3d.renderer.opengl import MeshRasterizerOpenGL
#=====================================faster ===========================
from pytorch3d.ops import interpolate_face_attributes
import pdb

from omegaconf import OmegaConf
import warnings
import math

CUDA_LAUNCH_BLOCKING = 1
warnings.filterwarnings('ignore')

#========================================================================================
# pytorch3d renderer
#========================================================================================


class Render:

    def __init__(self, device, obj_catergory, num_envs, endground, conf_file,
                 objects_mesh, num_cameras, obj_names, image_size, task_index):

        # load conf files
        self.conf = OmegaConf.load(conf_file)

        self.objects_mesh = objects_mesh

        self.obj_catergory = obj_catergory

        self.num_cameras = num_cameras  # number of cameras in per robot

        self.device = device  #cuda or cpu(it is better to use cuda)

        self.batch_size = self.num_cameras * num_envs  # maximum number of cameras in the env

        self.obj_names = obj_names

        self.image_size = 64  #image_size

        # init mesh
        self.init_mesh()  # init gel mesh(digit surface)

        # init camera
        self.init_camera()  # init camera parameter in the pyrender

        # init light
        self.init_light()  # init RGB lights in the scene

        # init rasterizer
        self.init_rasterizer()  # init rasterizer for render

        # force  adjustment(clamp by min and max force threshold)
        if self.conf.sensor.force.enable:
            self.min_force = self.conf.sensor.force.range_force[0]
            self.max_force = self.conf.sensor.force.range_force[1]
            if task_index == 0:  #insertion
                self.max_deformation = self.conf.sensor.force.max_deformation
            elif task_index == 1:  #twisting and grasping

                self.max_deformation = 0.001

            elif task_index == 2:  #extrinsic
                self.max_deformation = -0.01

        # endground image get from real digit
        self.endground = torch.as_tensor(cv2.resize(endground,
                                                    (self.height, self.width)),
                                         device=self.device)

    # init gel mesh(refer from tacto,carefully change paras)
    #========================================================================================
    def init_mesh(self) -> None:

        # Load config
        g = self.conf.sensor.gel
        origin = g.origin

        X0, Y0, Z0 = origin[0], origin[1], origin[2]
        W, H = g.width, g.height

        N = g.countW

        M = int(N * H / W)

        R = g.R
        zrange = g.curvatureMax

        y = np.linspace(Y0 - W / 2, Y0 + W / 2, N)
        z = np.linspace(Z0 - H / 2, Z0 + H / 2, M)
        yy, zz = np.meshgrid(y, z)

        h = R - np.maximum(0, R**2 - (yy - Y0)**2 - (zz - Z0)**2)**0.5

        xx = X0 - zrange * h / h.max()

        self.generate_trimesh_from_depth(xx)

    #========================================================================================
    # generate trimesh(refer from tacto)
    #========================================================================================
    def generate_trimesh_from_depth(self, depth: float) -> None:
        # Load config
        g = self.conf.sensor.gel
        origin = g.origin

        _, Y0, Z0 = origin[0], origin[1], origin[2]
        W, H = g.width, g.height

        N = depth.shape[1]
        M = depth.shape[0]

        # Create grid mesh
        vertices = []
        faces = []

        y = np.linspace(Y0 - W / 2, Y0 + W / 2, N)
        z = np.linspace(Z0 - H / 2, Z0 + H / 2, M)
        yy, zz = np.meshgrid(y, z)

        vertices = np.zeros([N * M, 3])

        # Add x, y, z position to vertex
        vertices[:, 0] = depth.reshape([-1])
        vertices[:, 1] = yy.reshape([-1])
        vertices[:, 2] = zz.reshape([-1])
        self.digit_vertices = torch.as_tensor(vertices,
                                              device=self.device,
                                              dtype=torch.float32)

        # Create faces

        faces = torch.zeros([(N - 1) * (M - 1) * 6],
                            dtype=torch.int,
                            device=self.device)

        # calculate id for each vertex: (i, j) => i * m + j
        xid = torch.arange(N)
        yid = torch.arange(M)
        yyid, xxid = torch.meshgrid(xid, yid)

        ids = yyid[:-1, :-1].reshape([-1]) + xxid[:-1, :-1].reshape([-1]) * N

        # create upper triangle
        faces[::6] = ids  # (i, j)
        faces[1::6] = ids + N  # (i+1, j)
        faces[2::6] = ids + 1  # (i, j+1)

        # create lower triangle
        faces[3::6] = ids + 1  # (i, j+1)
        faces[4::6] = ids + N  # (i+1, j)
        faces[5::6] = ids + N + 1  # (i+1, j+1)

        self.digit_faces = faces.view([-1, 3])

    #========================================================================================
    # init lighting(refer from tacto)
    #========================================================================================
    def init_light(self) -> None:
        """
        init the light information
       """
        # transparancy
        self.alphas = torch.ones((self.batch_size, self.height, self.width),
                                 device=self.device) * 1.0
        # light position
        # Load light from config file
        light = self.conf.sensor.lights

        origin = np.array(light.origin, dtype=np.double)

        xyz = []
        if light.polar:
            # Apply polar coordinates
            thetas = light.xrtheta.thetas
            rs = light.xrtheta.rs
            xs = light.xrtheta.xs
            for i in range(len(thetas)):
                theta = np.pi / 180 * thetas[i]
                xyz.append(
                    [xs[i], rs[i] * np.cos(theta), rs[i] * np.sin(theta)])

        else:
            # Apply cartesian coordinates
            xyz = np.array(light.xyz.coords)

        translation = np.array(xyz + origin)
        translation = torch.as_tensor(translation,
                                      device=self.device,
                                      dtype=torch.float32)

        self.light_positions0 = torch.unsqueeze(translation, dim=0)

        # light color
        light_intensities = torch.as_tensor(light.colors,
                                            device=self.device) * 1.0
        self.light_intensities0 = torch.unsqueeze(light_intensities, dim=0)

        self.light_intensities = torch.repeat_interleave(
            self.light_intensities0, self.batch_size + 100, dim=0)

        # diffuse color
        self.diffuse_colors = torch.ones(
            (self.batch_size + 100, self.height, self.width, 3),
            device=self.device)

        # specular_colors
        self.specular_colors = self.diffuse_colors

        # shininess_coefficients
        self.shininess_coefficients = torch.ones(
            (self.batch_size + 100, self.height, self.width),
            device=self.device)

        # ambient_color
        self.ambient_color = torch.ones((self.batch_size + 100, 3),
                                        device=self.device)

        self.light = light

    #========================================================================================
    # init pytorch3d camera
    #========================================================================================
    def init_camera(self) -> None:

        # The transformation matrix between the pytorch3d coordinate and orignial digit cameras coordinate
        self.flip_R = torch.unsqueeze(torch.as_tensor(
            [[6.1232e-17, 0.0000e+00, 1.0000e+00],
             [1.0000e+00, 6.1232e-17, -6.1232e-17],
             [-6.1232e-17, 1.0000e+00, 3.7494e-33]],
            device=self.device),
                                      dim=0)

        self.flip_quat = (matrix_to_quaternion(self.flip_R)
                          )  # rotation to quaternion

        # init the cameras postion
        # the digit cameras is at (0,0,0.015), the same location in the pytorch3d is (0,0,-0.015)
        init_camera_pos = torch.unsqueeze(torch.as_tensor([0., 0, -0.015],
                                                          device=self.device),
                                          dim=0)

        # prepare for transformation
        transform = Transform3d(device=self.device).rotate(self.flip_R)
        # the camera location in the gel perspective
        self.init_camera_pos = transform.transform_points(
            init_camera_pos.view(-1, 3))

        # init size information
        # self.image_size, self.height, self.width = self.conf.sensor.camera.image_size, self.conf.sensor.camera.height, self.conf.sensor.camera.width
        self.height = self.image_size
        self.width = self.image_size
        # self.cameras = OpenGLPerspectiveCameras(device=self.device,
        #                                         znear=0.001,
        #                                         aspect_ratio=3 / 4,
        #                                         fov=60,
        #                                         R=self.flip_R,
        #                                         T=self.init_camera_pos)

        # self.cameras_center = self.cameras.get_camera_center()

        # self.rasterizer = MeshRasterizer(cameras=self.cameras,
        #                                  raster_settings=self.raster_settings)

    #========================================================================================
    # init rasterizer setting
    #========================================================================================
    def init_rasterizer(self) -> None:
        '''
        the params are free to be changed
        '''

        self.raster_settings = RasterizationSettings(
            image_size=self.image_size,
            # max_faces_per_bin=3000
            # faces_per_pixel=1,
        )

    #========================================================================================
    # render image without touch
    #========================================================================================
    def render_static(self) -> None:
        '''
        render the image without any force
        '''

        static_mesh = Meshes(verts=[self.digit_vertices],
                             faces=[self.digit_faces],
                             textures=None)

        camera_quat0 = torch.as_tensor(
            [1, 0, 0, 0.0],
            device=self.device,
        ).view(-1, 4)
        camera_pos0 = torch.as_tensor(
            [0, 0, 0],
            device=self.device,
        ).view(-1, 3)

        self.update_camera(camera_pos0, camera_quat0)

        self.cameras = OpenGLPerspectiveCameras(device=self.device,
                                                znear=0.001,
                                                aspect_ratio=3 / 4,
                                                fov=60,
                                                R=self.R_cameras,
                                                T=self.T_cameras)

        self.rasterizer = MeshRasterizer(cameras=self.cameras,
                                         raster_settings=self.raster_settings)

        fragments = self.rasterizer(static_mesh)

        pix_to_face, depth, bary_coords = fragments.pix_to_face, fragments.zbuf, fragments.bary_coords

        # compute the pixel normal and coord
        pixel_normals, pixel_coords = self.compute_pixel_normal_coord(
            static_mesh, pix_to_face, bary_coords)

        # normalize pixel normal
        pixel_normals = pixel_normals.view(-1, self.height, self.width, 3)
        pixel_normals = pixel_normals / \
            torch.norm(pixel_normals, dim=3, keepdim=True)

        # resize pixel coord
        pixel_coords = pixel_coords.view(-1, self.height, self.width, 3)

        # render image by phong shader model
        rgb = self.phong_shader(
            pixel_normals, self.alphas[0].view(-1, self.width,
                                               self.height), pixel_coords,
            self.light_positions0, self.light_intensities[0].view(-1, 3, 3),
            self.diffuse_colors[0].view(-1, self.width, self.height, 3))

        self.static_rgb = rgb
        self.depth0 = depth[0]

        # print(self.static_rgb.shape)
        # cv2.imwrite(
        #     '/media/lme/data/sapien_task/conf/static.png',
        #     cv2.cvtColor((self.static_rgb[0][:, :, :3].detach().cpu().numpy() *
        #                   255).astype(np.uint8), cv2.COLOR_RGB2BGR))
        return self.static_rgb, self.depth0

    #========================================================================================
    # update lights position
    #========================================================================================
    def update_light(self, cameras_pos: torch.Tensor,
                     cameras_quat: torch.Tensor) -> None:
        '''
        update the position of light (Not Change)
        '''
        self.alphas = torch.ones((self.batch_size, self.height, self.width),
                                 device=self.device)

        self.light_positions = self.light_positions0

        self.light_intensities = torch.repeat_interleave(
            self.light_intensities0, self.batch_size, dim=0)

        # diffuse color
        self.diffuse_colors = torch.ones(
            (self.batch_size, self.height, self.width, 3), device=self.device)

        # specular_colors
        self.specular_colors = self.diffuse_colors

        # shininess_coefficients
        self.shininess_coefficients = torch.ones(
            (self.batch_size, self.height, self.width), device=self.device)

        # ambient_color
        self.ambient_color = torch.ones((self.batch_size, 3),
                                        device=self.device)

    #========================================================================================
    # update gel vertices
    #========================================================================================
    def update_gel_vertices(self, cameras_pos: torch.Tensor,
                            cameras_quat: torch.Tensor) -> torch.Tensor:
        '''
        update the vertcies for the gel mesh (Not change)
        '''

        gel_mesh = torch.repeat_interleave(self.digit_vertices.view(1, -1, 3),
                                           len(cameras_pos),
                                           dim=0)

        return gel_mesh

    #========================================================================================
    # adjust the postion by forces
    #========================================================================================
    def adjust_with_force(self, obj_pos: torch.Tensor,
                          cameras_pos: torch.Tensor,
                          robo_force: torch.Tensor) -> torch.Tensor:

        max_force = torch.as_tensor(self.max_force, device=self.device).expand(
            robo_force.view(-1).shape)

        offset = (torch.min(max_force, robo_force.view(-1)) / self.max_force
                  )  #* 0 + 1

        direction = obj_pos[:, -1] - cameras_pos

        if self.obj_catergory in ["USB"]:
            self.max_deformation = 0



        direction = direction / \
            (torch.sum(direction ** 2, dim=1, keepdim=True) ** 0.5 + 1e-6)
        max_deformation = self.max_deformation


        cameras_pos = cameras_pos + \
            offset.view(-1, 1) * max_deformation * direction

        return cameras_pos

    #========================================================================================
    # update the postion of cameras
    #========================================================================================
    def update_camera(self, cameras_pos: torch.Tensor,
                      cameras_quat: torch.Tensor):
        '''
        very tricky transformation
        '''

        # add flip R to the rotation to flip the camera
        self.R_cameras = torch.repeat_interleave(self.flip_R,
                                                 len(cameras_pos),
                                                 dim=0)

        # obtain T_cameras
        self.T_cameras = torch.repeat_interleave(self.init_camera_pos,
                                                 len(cameras_pos),
                                                 dim=0)

        # pdb.set_trace()

        # self.cameras = OpenGLPerspectiveCameras(device=self.device,
        #                                         znear=0.001,
        #                                         aspect_ratio=3 / 4,
        #                                         fov=60,
        #                                         R=self.R_cameras,
        #                                         T=self.T_cameras)

        # self.cameras_center = self.cameras.get_camera_center()

        # self.rasterizer = MeshRasterizer(cameras=self.cameras,
        #                                  raster_settings=self.raster_settings)

    #========================================================================================
    # update the vertices of object
    #========================================================================================
    def update_obj_vertices(self, cameras_pos: torch.Tensor,
                            cameras_quat: torch.Tensor, obj_pos: torch.Tensor,
                            obj_quat: torch.Tensor) -> torch.Tensor:
        '''
        To simplify the calculation, assume the location and orientation of digit
        as unchanged and the relative transformation matrix between the digit and object
        need to be calculated
        
        '''

        if obj_quat.shape[1] == 1:
            obj_mesh, obj_face = self.update_single_link_vertices(
                cameras_pos, cameras_quat, obj_pos[:, 0], obj_quat[:, 0])

        else:
            obj_mesh, obj_face = self.update_multi_links_vertices(
                cameras_pos, cameras_quat, obj_pos, obj_quat)

        return obj_mesh, obj_face

    def update_transform(self, cameras_pos: torch.Tensor,
                         cameras_quat: torch.Tensor, obj_pos: torch.Tensor,
                         obj_quat: torch.Tensor) -> torch.Tensor:

        #obtain quaternion for the relative transformation
        quaternion = quaternion_multiply(quaternion_invert(cameras_quat),
                                         obj_quat)
        rotation = quaternion_to_matrix(quaternion_invert(quaternion))

        #obtain translations for the relative transformation
        transform = Transform3d(device=self.device).rotate(
            quaternion_to_matrix((cameras_quat)))
        translations = transform.transform_points(
            (obj_pos - cameras_pos).view(-1, 1, 3))[:, 0]

        # apply transformations to vertices
        transform = Transform3d(
            device=self.device).rotate(rotation).translate(translations)

        return transform

    def update_multi_links_vertices(self, cameras_pos: torch.Tensor,
                                    cameras_quat: torch.Tensor,
                                    obj_pos: torch.Tensor,
                                    obj_quat: torch.Tensor) -> torch.Tensor:

        obj_meshes = []
        obj_face = []

        multi_link_meshes = []

        if isinstance(self.obj_index, int):
            self.obj_index = [self.obj_index]
        for i, obj_index in enumerate(self.obj_index):
            multi_mesh = []
            obj_face.append(
                torch.cat([
                    self.digit_faces, self.objects_mesh[obj_index][1] +
                    self.digit_vertices.shape[0]
                ],
                          dim=0))
            multi_link_meshes.append(self.objects_mesh[obj_index][0])

            transform = self.update_transform(cameras_pos[i], cameras_quat[i],
                                              obj_pos[i, :, :],
                                              obj_quat[i, :, :])

            for j in range(obj_pos.shape[1]):

                scales = self.obj_scales[j]

                scale_vertices = self.objects_mesh[obj_index][0][j] * scales

                obj_mesh = transform[j].transform_points(scale_vertices)

                multi_mesh.append(obj_mesh)

            obj_meshes.append(
                torch.cat(
                    [self.digit_vertices,
                     torch.concat(multi_mesh, dim=0)],
                    dim=0))

        return obj_meshes, obj_face

    def update_single_link_vertices(self, cameras_pos: torch.Tensor,
                                    cameras_quat: torch.Tensor,
                                    obj_pos: torch.Tensor,
                                    obj_quat: torch.Tensor) -> torch.Tensor:

        # #obtain quaternion for the relative transformation
        # quaternion = quaternion_multiply(quaternion_invert(cameras_quat),
        #                                  obj_quat)
        # rotation = quaternion_to_matrix(quaternion_invert(quaternion))

        # #obtain translations for the relative transformation
        # transform = Transform3d(device=self.device).rotate(
        #     quaternion_to_matrix((cameras_quat)))
        # translations = transform.transform_points(
        #     (obj_pos - cameras_pos).view(-1, 1, 3))[:, 0]

        # # apply transformations to vertices
        # transform = Transform3d(
        #     device=self.device).rotate(rotation).translate(translations)
        transform = self.update_transform(cameras_pos, cameras_quat, obj_pos,
                                          obj_quat)

        # cameras_transform = Transform3d(device=self.device).rotate(
        #     quaternion_to_matrix((cameras_quat)))
        # cameras_translations = Transform3d.transform_points(
        #     (obj_pos - cameras_pos).view(-1, 1, 3))[:, 0]

        # apply transformations to vertices
        # cameras_transform = Transform3d(device=self.device).rotate(
        #     quaternion_to_matrix((cameras_quat))).translate(cameras_pos)
        # obj_transform = Transform3d(device=self.device).rotate(
        #     quaternion_to_matrix((cameras_quat))).translate(cameras_pos)

        obj_mesh = []
        obj_face = []

        # if isinstance(self.obj_index, int):

        #     obj_mesh = transform.transform_points(
        #         self.objects_mesh[self.obj_index][0].view(1, -1, 3))
        #     obj_face = self.objects_mesh[self.obj_index][1]

        # else:
        #     # print("rendering obj check:")

        for i, obj_index in enumerate(self.obj_index):

            obj_index = int(obj_index)

            scales = self.obj_scales[i]

            scale_vertices = self.objects_mesh[obj_index][0][0] * (scales)

            transform_obj_mesh = transform[i].transform_points(
                scale_vertices.view(1, -1, 3))[0]

            obj_mesh.append(
                torch.cat([self.digit_vertices, transform_obj_mesh], dim=0))

            obj_face.append(
                torch.cat([
                    self.digit_faces, self.objects_mesh[obj_index][1] +
                    self.digit_vertices.shape[0]
                ],
                          dim=0))

            # obj_mesh.append(transform_obj_mesh)

            # obj_face.append(self.objects_mesh[obj_index][1])
        # ax = plt.axes(projection='3d')
        # # points_pixel = pixel_coords[0].cpu().view(-1, 1, 3)
        # # ax.scatter(points_pixel[:, :, 0].reshape(-1),
        # #            points_pixel[:, :, 1].reshape(-1),
        # #            points_pixel[:, :, 2].reshape(-1),
        # #            c=points_pixel[:, :, 0].reshape(-1))

        # points_pixel = self.objects_mesh[obj_index][0][0].cpu().view(-1, 1, 3)
        # ax.scatter(points_pixel[:, :, 0].reshape(-1),
        #            points_pixel[:, :, 1].reshape(-1),
        #            points_pixel[:, :, 2].reshape(-1),
        #            c='r')
        # # points_pixel = self.digit_vertices.cpu().view(-1, 1, 3)
        # # ax.scatter(points_pixel[:, :, 0].reshape(-1),
        # #            points_pixel[:, :, 1].reshape(-1),
        # #            points_pixel[:, :, 2].reshape(-1),
        # #            c='b')
        # plt.show()

        return obj_mesh, obj_face

    #========================================================================================
    # join gel and object into one scene
    #========================================================================================
    def join_mesh_scene(self, obj_mesh: List[torch.Tensor],
                        obj_face: List[torch.Tensor],
                        gel_mesh: torch.Tensor) -> torch.Tensor:
        '''
        joint obj and gel into one scene
        '''

        verts1, verts2 = gel_mesh, obj_mesh
        faces1, face2 = self.digit_faces, obj_face

        if isinstance(self.obj_index, int):

            faces1 = torch.repeat_interleave(
                faces1.view(1, -1, 3), len(verts1),
                dim=0)  # number of scence need to render
            face2 = torch.repeat_interleave(
                face2.view(1, -1, 3), len(verts1),
                dim=0)  # number of scence need to render

            verts = torch.cat([verts1, verts2], dim=1)

            #  Offset by the number of vertices in mesh1
            face2 = face2 + verts1.shape[1]

            faces = torch.cat([faces1, face2], dim=1)  # (400, 3)
            return verts, faces

        else:

            return obj_mesh, obj_face

        # return verts2.to(self.device), faces2.to(self.device)

    #========================================================================================
    # compute the coord and normal for pixel
    #========================================================================================
    def compute_pixel_normal_coord(
            self, mesh, pix_to_face: torch.Tensor,
            bary_coords: torch.Tensor) -> Union[torch.Tensor, torch.Tensor]:
        '''
        compute the normal vector and coordinate for each pixel in the image
        '''

        # compute pixel normal
        faces = mesh.faces_packed()  # (F, 3)
        vertex_normals = mesh.verts_normals_packed()  # (V, 3)
        faces_normals = vertex_normals[faces]
        pixel_normals = interpolate_face_attributes(pix_to_face, bary_coords,
                                                    faces_normals)

        # compute pixel coord
        verts_packed = mesh.verts_packed()
        faces_verts = verts_packed[faces]
        pixel_coords = interpolate_face_attributes(pix_to_face, bary_coords,
                                                   faces_verts)
        return pixel_normals, pixel_coords

    #========================================================================================
    # render mesh as batch
    #========================================================================================

    def render_mesh_batch(
            self, verts_list: List, faces_list: List
    ) -> Union[torch.Tensor, torch.Tensor, torch.Tensor]:
        '''
        render the mesh as batch to reduce GPU storage
        
        Input:  verts_list,
                faces_list,
                start(the starter index for the vertices)
                end(the end index for the vertices)
        '''

        mesh = Meshes(verts=verts_list, faces=faces_list, textures=None)
        self.cameras = OpenGLPerspectiveCameras(device=self.device,
                                                znear=0.001,
                                                aspect_ratio=3 / 4,
                                                fov=60,
                                                R=self.R_cameras,
                                                T=self.T_cameras)

        # Init rasterizer

        self.rasterizer = MeshRasterizer(cameras=self.cameras,
                                         raster_settings=self.raster_settings)

        # render

        fragments = self.rasterizer(mesh)

        pix_to_face, depth, bary_coords = fragments.pix_to_face, fragments.zbuf, fragments.bary_coords

        # obtain normnals and coords for each pixel in the image
        pixel_normals, pixel_coords = self.compute_pixel_normal_coord(
            mesh, pix_to_face, bary_coords)

        return pixel_normals, pixel_coords, depth

    def update(
            self, obj_pos: torch.Tensor, obj_quat: torch.Tensor,
            cameras_pos: torch.Tensor,
            cameras_quat: torch.Tensor) -> Union[torch.Tensor, torch.Tensor]:
        '''
        update light, cameras, mesh
        '''
        # update lights location
        self.update_light(cameras_pos, cameras_quat)
        # update cameras location
        self.update_camera(cameras_pos, cameras_quat)

        #update the mesh vertices of the objects
        obj_mesh, obj_face = self.update_obj_vertices(cameras_pos,
                                                      cameras_quat, obj_pos,
                                                      obj_quat)
        #update the mesh vertices of the gel
        gel_mesh = self.update_gel_vertices(cameras_pos, cameras_quat)

        return obj_mesh, obj_face, gel_mesh

    #========================================================================================
    # render mesh
    #========================================================================================
    def render_mesh(
        self, obj_pos: torch.Tensor, obj_quat: torch.Tensor,
        cameras_pos: torch.Tensor, cameras_quat: torch.Tensor
    ) -> Union[torch.Tensor, torch.Tensor, torch.Tensor]:
        '''
        render mesh 
        '''

        obj_mesh, obj_face, gel_mesh = self.update(obj_pos, obj_quat,
                                                   cameras_pos, cameras_quat)
        #joint mesh into one scene
        # verts_list, faces_list = self.join_mesh_scene(obj_mesh, obj_face,
        #                                               gel_mesh)

        # pixel_normals, pixel_coords, depth = self.render_mesh_batch(
        #     verts_list, faces_list)

        pixel_normals, pixel_coords, depth = self.render_mesh_batch(
            obj_mesh, obj_face)

        # ax = plt.axes(projection='3d')
        # # points_pixel = pixel_coords[0].cpu().view(-1, 1, 3)
        # # ax.scatter(points_pixel[:, :, 0].reshape(-1),
        # #            points_pixel[:, :, 1].reshape(-1),
        # #            points_pixel[:, :, 2].reshape(-1),
        # #            c=points_pixel[:, :, 0].reshape(-1))
        # points_pixel = obj_mesh[-1].cpu().view(-1, 1, 3)
        # ax.scatter(points_pixel[:, :, 0].reshape(-1),
        #            points_pixel[:, :, 1].reshape(-1),
        #            points_pixel[:, :, 2].reshape(-1),
        #            c='r')
        # points_pixel = gel_mesh[-1].cpu().view(-1, 1, 3)
        # ax.scatter(points_pixel[:, :, 0].reshape(-1),
        #            points_pixel[:, :, 1].reshape(-1),
        #            points_pixel[:, :, 2].reshape(-1),
        #            c='b')
        # plt.show()

        return pixel_normals, pixel_coords, depth

    #========================================================================================
    # phong shader model(https://github.com/google/tf_mesh_renderer)
    #========================================================================================
    def phong_shader(self,
                     normals: torch.Tensor,
                     alphas: torch.Tensor,
                     pixel_positions: torch.Tensor,
                     light_positions: torch.Tensor,
                     light_intensities: torch.Tensor,
                     diffuse_colors=None,
                     camera_position=None,
                     specular_colors=None,
                     shininess_coefficients=None,
                     ambient_color=None) -> Union[torch.Tensor, torch.Tensor]:
        """Computes pixelwise lighting from rasterized buffers with the Phong model.
        `Args:
        normals: a 4D float32 tensor with shape [batch_size, image_height,
            image_width, 3]. The inner dimension is the world space XYZ normal for
            the corresponding pixel. Should be already normalized.
        alphas: a 3D float32 tensor with shape [batch_size, image_height,
            image_width]. The inner dimension is the alpha value (transparency)
            for the corresponding pixel.
        pixel_positions: a 4D float32 tensor with shape [batch_size, image_height,
            image_width, 3]. The inner dimension is the world space XYZ position for
            the corresponding pixel.
        light_positions: a 3D tensor with shape [batch_size, light_count, 3]. The
            XYZ position of each light in the scene. In the same coordinate space as
            pixel_positions.
        light_intensities: a 3D tensor with shape [batch_size, light_count, 3]. The
            RGB intensity values for each light. Intensities may be above one.
        diffuse_colors: a 4D float32 tensor with shape [batch_size, image_height,
            image_width, 3]. The inner dimension is the diffuse RGB coefficients at
            a pixel in the range [0, 1].
        camera_position: a 1D tensor with shape [batch_size, 3]. The XYZ camera
            position in the scene. If supplied, specular reflections will be
            computed. If not supplied, specular_colors and shininess_coefficients
            are expected to be None. In the same coordinate space as
            pixel_positions.
        specular_colors: a 4D float32 tensor with shape [batch_size, image_height,
            image_width, 3]. The inner dimension is the specular RGB coefficients at
            a pixel in the range [0, 1]. If None, assumed to be tf.zeros()
        shininess_coefficients: A 3D float32 tensor that is broadcasted to shape
            [batch_size, image_height, image_width]. The inner dimension is the
            shininess coefficient for the object at a pixel. Dimensions that are
            constant can be given length 1, so [batch_size, 1, 1] and [1, 1, 1] are
            also valid input shapes.
        ambient_color: a 2D tensor with shape [batch_size, 3]. The RGB ambient
            color, which is added to each pixel before tone mapping. If None, it is
            assumed to be tf.zeros().
        Returns:
        A 4D float32 tensor of shape [batch_size, image_height, image_width, 4]
        containing the lit RGBA color values for each image at each pixel. Colors
        are in the range [0,1].
        Raises:
        ValueError: An invalid argument to the method is detected.
        """

        batch_size, image_height, image_width = [s for s in normals.shape[:-1]]
        light_count = light_positions.shape[1]
        pixel_count = image_height * image_width
        normals = normals.view(batch_size, -1, 3)
        alphas = alphas.view(batch_size, -1, 1)
        diffuse_colors = diffuse_colors.view(batch_size, -1, 3)

        if camera_position is not None:
            specular_colors = specular_colors.view(batch_size, -1, 3)

        output_colors = torch.zeros(
            [batch_size, image_height * image_width, 3], device=self.device)
        if ambient_color is not None:
            ambient_reshaped = torch.unsqueeze(ambient_color, axis=1)
            output_colors = torch.add(output_colors,
                                      ambient_reshaped * diffuse_colors)

        # Diffuse component
        pixel_positions = pixel_positions.view(batch_size, -1, 3)
        per_light_pixel_positions = torch.stack(
            [pixel_positions] * light_count,
            axis=1)  # [batch_size, light_count, pixel_count, 3]
        directions_to_lights = torch.nn.functional.normalize(
            torch.unsqueeze(light_positions, axis=2) -
            per_light_pixel_positions,
            dim=3)  # [batch_size, light_count, pixel_count, 3]

        # The specular component should only contribute when the light and normal
        # face one another (i.e. the dot product is nonnegative):
        normals_dot_lights = torch.clamp(
            torch.sum(torch.unsqueeze(normals, axis=1) * directions_to_lights,
                      axis=3), 0.0,
            1.0)  # [batch_size, light_count, pixel_count]
        diffuse_output = torch.unsqueeze(
            diffuse_colors, axis=1) * torch.unsqueeze(
                normals_dot_lights, axis=3) * torch.unsqueeze(
                    light_intensities, axis=2)
        diffuse_output = torch.sum(diffuse_output,
                                   axis=1)  # [batch_size, pixel_count, 3]
        output_colors = torch.add(output_colors, diffuse_output)

        # Specular component
        if camera_position is not None:
            camera_position = camera_position.view(batch_size, 1, 3)
            mirror_reflection_direction = torch.nn.functional.normalize(
                2.0 * torch.unsqueeze(normals_dot_lights, axis=3) *
                torch.unsqueeze(normals, axis=1) - directions_to_lights,
                dim=3)
            direction_to_camera = torch.nn.functional.normalize(
                camera_position - pixel_positions, dim=2)
            reflection_direction_dot_camera_direction = torch.sum(
                torch.unsqueeze(direction_to_camera, axis=1) *
                mirror_reflection_direction,
                axis=3)
            # The specular component should only contribute when the reflection is
            # external:
            reflection_direction_dot_camera_direction = torch.clamp(
                torch.nn.functional.normalize(
                    reflection_direction_dot_camera_direction, dim=2), 0.0,
                1.0)
            # The specular component should also only contribute when the diffuse
            # component contributes:
            reflection_direction_dot_camera_direction = torch.where(
                normals_dot_lights != 0.0,
                reflection_direction_dot_camera_direction,
                torch.zeros_like(reflection_direction_dot_camera_direction,
                                 dtype=torch.float32))
            # Reshape to support broadcasting the shininess coefficient, which rarely
            # varies per-vertex:
            reflection_direction_dot_camera_direction = reflection_direction_dot_camera_direction.view(
                batch_size, light_count, image_height, image_width)
            shininess_coefficients = torch.unsqueeze(shininess_coefficients,
                                                     axis=1)
            specularity = torch.pow(reflection_direction_dot_camera_direction,
                                    shininess_coefficients).view(
                                        batch_size, light_count, pixel_count,
                                        1)
            specular_output = torch.unsqueeze(
                specular_colors, axis=1) * specularity * torch.unsqueeze(
                    light_intensities, axis=2)
            specular_output = torch.sum(specular_output, axis=1)
            output_colors = torch.add(output_colors, specular_output)
        rgb_images = output_colors.view(batch_size, image_height, image_width,
                                        3)

        alpha_images = alphas.view(batch_size, image_height, image_width, 1)
        valid_rgb_values = torch.cat(3 * [alpha_images > 0.5], axis=3)
        rgb_images = torch.where(
            valid_rgb_values, rgb_images,
            torch.zeros_like(rgb_images, dtype=torch.float32))
        return torch.cat([rgb_images, alpha_images], axis=3)

    def calibration_bg(self, rgb: torch.Tensor):

        diff = rgb - self.static_rgb

        cali_rgb = self.endground + 0.3 * diff[:, :, :, :3]
        return diff[:, :, :, :3], cali_rgb

    #========================================================================================
    # render
    #========================================================================================
    def render(self, obj_pos: torch.Tensor, obj_quat: torch.Tensor,
               cameras_pos: torch.Tensor, cameras_quat: torch.Tensor,
               robo_force: torch.Tensor, obj_index: int,
               obj_scales: torch.Tensor) -> Union[torch.Tensor, torch.Tensor]:

        # ajust the camera pos by force

        self.batch_size = len(obj_pos)
        self.obj_index = obj_index
        self.obj_scales = obj_scales

        if (obj_pos is not None) and (
                self.conf.sensor.force.enable):  # only if we need to do this
            cameras_pos = self.adjust_with_force(obj_pos, cameras_pos,
                                                 robo_force)

        # render  batch

        pixel_normals, pixel_coords, depth = self.render_mesh(
            obj_pos, obj_quat, cameras_pos, cameras_quat)

        # normalize pixel normal
        pixel_normals = pixel_normals.view(self.batch_size, self.height,
                                           self.width, 3)
        pixel_normals = pixel_normals / \
            torch.norm(pixel_normals, dim=3, keepdim=True)

        # resize pixel coord
        pixel_coords = pixel_coords.view(self.batch_size, self.height,
                                         self.width, 3)

        # render image by phong shader model
        # print(torch.unique(pixel_normals[:, :, :, 2]))

        rgb = self.phong_shader(
            pixel_normals,
            self.alphas,
            pixel_coords,
            self.light_positions,
            self.light_intensities[:self.batch_size],
            self.diffuse_colors[:self.batch_size],
            # self.cameras_center,
            # self.specular_colors[:self.batch_size],
            # self.shininess_coefficients[:self.batch_size],
            # self.ambient_color[:self.batch_size]
        )

        depth = self.depth0 - depth

        if self.conf.sensor.camera.calibration:
            diff_image, cali_rgb = self.calibration_bg(rgb)

            if self.conf.sensor.camera.difference_img:
                return diff_image, depth
            if self.conf.sensor.camera.caliberation_rgb:
                return cali_rgb, depth
        torch.cuda.empty_cache()

        # pixel_normals = pixel_normals / pixel_normals[:, :, :, 2][:, :, :,
        #                                                           None] * -1
        # return pixel_normals, depth

        return rgb[:, :, :, :3], depth


######################################################################################################################################
