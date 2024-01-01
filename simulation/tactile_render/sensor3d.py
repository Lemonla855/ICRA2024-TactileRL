from re import I
import yaml
# import Sensor
# import matplotlib.pyplot as plt
from PIL import Image as im

import copy
import pdb
# import torch
from urdfpy import URDF
from scipy.spatial.transform import Rotation as R
from typing import Union
import math
import torch
# from tacto import Sensor
import numpy as np
import time
import cv2

import trimesh

from hand_teleop.tactile_render.render3d import Render


#==============================================================
# init pytorch3d sensor
#==============================================================
class Sensor():
    def __init__(self,
                 device,
                 conf_file,
                 num_env,
                 obj,
                 robo_tactile_link,
                 num_gel,
                 obj_urdf_paths,
                 globalScalings,
                 scene,
                 background,
                 visualize_gui=True,
                 show_depth=True):

        self.device = device
        self.robo_tactile_link = robo_tactile_link
        self.obj = obj

        # init visulization information
        self.visualize_gui = visualize_gui
        self.show_depth = show_depth
        self.scene = scene

        # init camera information
        self.cameras = {}

        self.num_env = num_env
        self.num_gel = num_gel
        self.objects_mesh = []

        # preload the mesh vertices to reduce the computation time
        self.add_object(obj_urdf_paths, globalScalings)

        self.renderer = Render(device, self.num_env, background, conf_file,
                               self.objects_mesh, num_gel)  #init renderer

        self.init_render_static()  # render static image in advance

        # self.index_dict()

        # cv2.imwrite('../conf/bg.png',
        #             cv2.cvtColor(
        #                 (self.rgb0[0].cpu().numpy() * 255).astype(np.uint8),
        #                 cv2.COLOR_RGB2BGR))  # save background image

    # #==============================================================
    # # create dictory for the index reference for digit
    # #==============================================================

    # def index_dict(self):

    #     self.digit_dict = {}

    #     for i, link in enumerate(self.robo_tactile_link):

    #         self.digit_dict[link.name] = i

    #==============================================================
    # render image without tounch
    #==============================================================
    def init_render_static(self) -> None:

        self.rgb, self.depth = self.renderer.render_static()

        self.rgb0 = torch.repeat_interleave(self.rgb[:, :, :, :3],
                                            len(self.robo_tactile_link),
                                            dim=0)

        self.depth0 = torch.repeat_interleave(torch.unsqueeze(self.depth * 0.0,
                                                              dim=0),
                                              len(self.robo_tactile_link),
                                              dim=0)

    # #==============================================================
    # # init index for reference
    # #==============================================================
    # def init_rigid_index(self, obj_index, robo_link_index):
    #     '''
    #     init the rigid index for the object and robot in the simulator
    #     '''

    #     obj_index = torch.as_tensor(
    #         obj_index, device=self.device)  # The index of contact object
    #     robo_link_index = torch.as_tensor(
    #         robo_link_index, device=self.device)  # The index of cameras
    #     self.objects_mesh = []

    #     # the index of the object in the simulation
    #     self.obj_sim_index = self.get_rigid_index(obj_index)
    #     self.obj_sim_index = self.obj_sim_index.repeat_interleave(self.num_gel)
    #     self.robo_tactile_link = self.get_rigid_index(robo_link_index)

    # #==============================================================
    # # init index for reference
    # #==============================================================
    # def get_rigid_index(self, rigid_index):
    #     '''
    #     preccalculate the index of obj/robot link in acquire_rigid_body_state_tensor to reduce time
    #     '''

    #     poses = self.gym.acquire_rigid_body_state_tensor(self.sim)

    #     poses = gymtorch.wrap_tensor(poses)
    #     num_dof_per_env = int(len(poses) / self.num_env)

    #     for env_index in range(self.num_env):
    #         if env_index == 0:
    #             rigid_index_envs = env_index * num_dof_per_env + rigid_index
    #         if env_index != 0:
    #             rigid_index_envs = torch.cat(
    #                 (rigid_index_envs,
    #                  env_index * num_dof_per_env + rigid_index),
    #                 dim=0)

    #     return rigid_index_envs

    #==============================================================
    #add object
    #==============================================================
    def add_object(self, urdf_fn: str, globalScaling: float = 1.0) -> None:
        '''
        pre load the mesh for rendering
        '''

        obj_trimesh = trimesh.load(urdf_fn)

        obj_trimesh.visual = trimesh.visual.ColorVisuals()

        vertices = torch.as_tensor(obj_trimesh.vertices,
                                   device=self.device,
                                   dtype=torch.float32)
        faces = torch.as_tensor(obj_trimesh.faces,
                                device=self.device,
                                dtype=torch.float32)

        self.objects_mesh.append([vertices, faces])

    # #==============================================================
    # # obtain contact force
    # #==============================================================
    # def obtain_force(self, ):
    #     '''
    #     obtain the force of the robot link
    #     '''
    #     actor_set1 = set(self.obj)
    #     actor_set2 = set(self.robo_tactile_link)

    #     contact_forces=[]
    #     contact_names=[]
    #     force_index=[]

    #     for contact in self.scene.get_contacts():
    #         contact_actors = {contact.actor0, contact.actor1}
    #         if len(actor_set1 & contact_actors) > 0 and len(actor_set2 & contact_actors) > 0:

    #             impulse = np.array([point.impulse for point in contact.points])
    #             contact_force=np.sum(np.linalg.norm(abs(impulse), axis=1)) / (1/2400)
    #             # print('===============================')
    #             # print(contact_actors)
    #             if contact_force<1e-2:
    #                 continue

    #             contact_forces.append(contact_force)

    #             if contact.actor0.name in ["link_3.0_gel","link_7.0_gel","link_11.0_gel","link_15.0_gel"]:
    #                  contact_names.append(contact.actor0)
    #                  force_index.append(self.digit_dict[contact.actor0.name])
    #             else:
    #                 contact_names.append(contact.actor1)
    #                 force_index.append(self.digit_dict[contact.actor1.name])

    #     return contact_names,contact_forces,force_index

    # #==============================================================
    # # update pose for all rigids
    # #==============================================================
    # def update_pose(self, contact_names):
    #     '''
    #     obtain the pos information for the related object according to the force info
    #     '''
    #     cameras_position=[]
    #     cameras_quat=[]
    #     obj_position=[]
    #     obj_quat=[]

    #     for contact_obj in contact_names:
    #         translation=contact_obj.get_pose().p
    #         quat=contact_obj.get_pose().q
    #         cameras_position.append([translation[0],translation[1],translation[2]])
    #         cameras_quat.append([quat[0],quat[1],quat[2],quat[3]])

    #     translation=self.obj[0].get_pose().p
    #     quat=self.obj[0].get_pose().q
    #     obj_position.append([translation[0],translation[1],translation[2]])
    #     obj_quat.append([quat[0],quat[1],quat[2],quat[3]])

    #     return cameras_position, cameras_quat, obj_position, obj_quat

    #==============================================================
    # numpy2torch
    #==============================================================

    def numpy2torch(self, data, repeat=False) -> torch.Tensor:

        data = torch.as_tensor(data, dtype=torch.float32, device=self.device)

        if repeat:
            data = torch.repeat_interleave(data, self.num_gel, dim=0)

        return data

    #==============================================================
    # render image
    #==============================================================
    def render(self, cameras_position, cameras_quat, obj_position, obj_quat,
               contact_forces,
               force_index) -> Union[torch.Tensor, torch.Tensor]:
        """
        Render tacto images from each camera's view.
        """
        start = time.time()

        # contact_names, contact_forces, force_index = self.obtain_force()

        output_color, out_depth = self.rgb0.clone(), self.depth0.clone()

        if not contact_forces:

            return output_color, out_depth

        # cameras_position, cameras_quat, obj_position, obj_quat = self.update_pose(
        #     contact_names)

        cameras_position = self.numpy2torch(cameras_position)
        cameras_quat = self.numpy2torch(cameras_quat)
        obj_position = self.numpy2torch(obj_position)
        obj_quat = self.numpy2torch(obj_quat)
        contact_forces = self.numpy2torch(contact_forces)

        # print("num of images", len(contact_forces))

        color, depth = self.renderer.render(
            obj_position,
            obj_quat,
            cameras_position,
            cameras_quat,
            contact_forces,
        )

        output_color[force_index] = color
        out_depth[force_index] = depth

        # force_index = force_index[0]

        # for i in range(num_batch):

        #     if i != num_batch - 1:
        #         front = per_batch * i
        #         back = per_batch * (i + 1)
        #         color, depth = self.renderer.render(
        #             obj_pos[front:back],
        #             obj_quat[front:back],
        #             cameras_pos[front:back],
        #             cameras_quat[front:back],
        #             robo_force[front:back],
        #         )

        #         output_color[force_index[front:back]] = color
        #         out_depth[force_index[front:back]] = depth
        #     else:
        #         front = per_batch * i
        #         back = None
        #         color, depth = self.renderer.render(
        #             obj_pos[front:],
        #             obj_quat[front:],
        #             cameras_pos[front:],
        #             cameras_quat[front:],
        #             robo_force[front:],
        #         )

        #         output_color[force_index[front:]] = color
        #         out_depth[force_index[front:]] = depth

        # output_color = output_color-self.rgb0
        end = time.time()
        # print("rendering per frame:", (end - start) / len(contact_forces))

        return output_color, out_depth

        # return output_color.cpu(), out_depth.cpu().numpy()

    def depth_to_color(self, depth, zrange) -> np.ndarray:
        "tactile depth modification to three channel images"

        gray = (np.clip(depth / zrange, 0, 1) * 255).astype(np.uint8)
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    #==============================================================
    # update render result for GUI if needed
    #==============================================================
    def updateGUI(self, colors, depths) -> None:
        """
        Update images for visualization
        """
        if not self.visualize_gui:
            return

        # concatenate colors horizontally (axis=1)
        color = np.concatenate(colors.cpu().numpy(), axis=1)

        if self.show_depth:

            # concatenate depths horizontally (axis=1)
            depth = self.depth_to_color(
                np.concatenate(depths.cpu().numpy(), axis=1), 0.002)

            # concatenate the resulting two images vertically (axis=0)

            color_n_depth = np.concatenate([(color), depth], axis=0)

            cv2.imshow("color and depth",
                       cv2.cvtColor(color_n_depth, cv2.COLOR_RGB2BGR))

        else:

            cv2.imshow("color", cv2.cvtColor(color, cv2.COLOR_RGB2BGR))
            #
            # cv2.imshow("color", color)

        cv2.waitKey(1)
