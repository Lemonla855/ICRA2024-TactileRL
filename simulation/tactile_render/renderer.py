from logging import captureWarnings
from hand_teleop.tactile_render.sensor3d import Sensor
from hand_teleop.tactile_render.sensor_tacto import Sensor_tacto, get_digit_config_path
from hand_teleop.tactile_render.Render_tacto import Renderer_tacto
from urdfpy import URDF
from scipy.spatial.transform import Rotation as R
import sapien.core as sapien
import torch
import numpy as np
import cv2
import imageio
import pdb
import sys
import time
from PIL import Image, ImageColor
from hand_teleop.env.rl_env.base import BaseRLEnv
from hand_teleop.env.sim_env.relocate_env import RelocateEnv, LabRelocateEnv


#==============================================================
# all renderer main entry
#==============================================================
class Renderer(RelocateEnv, BaseRLEnv):
    def __init__(
        self,
        device,  #cuda  or cpu (it is better to use cuda for pytorch3d rendering )
        conf_file,  # conf file for rendering 
        num_envs,  #number of envs in the simulation
        obj,  # the rigid body index of object in the sim
        robo_tactile_link,  # the rigid body index of tactile link of robot in the sim
        num_cams,  #num of cameras per robot
        globalScalings,  # the scale of the object(it is better to set as 1)
        urdf_files,  # urdf file of the object
        robot,  # handles of robot (for isaac view rendering)
        dt,  # number of time
        scene=None,
        visualize_gui=False,  # visulize gui for pytorch3d rendering only
        show_depth=True,  # show the depth info during visulization
        render_tacto=False,  # render the image by pyrender
        render_sapien_camera=False,  # render the image by isaac camera
        make_video=False):  #make video for visulization result

        super().__init__(
            device,  #cuda  or cpu (it is better to use cuda for pytorch3d rendering )
            conf_file,  # conf file for rendering 
            num_envs,  #number of envs in the simulation
            obj,  # the rigid body index of object in the sim
            robo_tactile_link,  # the rigid body index of tactile link of robot in the sim
            num_cams,  #num of cameras per robot
            globalScalings,  # the scale of the object(it is better to set as 1)
            urdf_files,  # urdf file of the object
            robot,  # handles of robot (for isaac view rendering)
            dt,  # number of time
            scene,
            visualize_gui,  # visulize gui for pytorch3d rendering only
            show_depth,  # show the depth info during visulization
            render_tacto,  # render the image by pyrender
            render_sapien_camera,  # render the image by isaac camera
            make_video)

        self.background_img = cv2.imread(
            "../conf/background.png")  # bacground img for real digit
        conf_file = '../conf/rendering.yaml'  # rendering config file for pytorch3d rendering

        # init rendering info
        self.robo_tactile_link = robo_tactile_link
        self.obj_tactile_link = obj
        self.num_cams = num_cams
        self.urdf_files = urdf_files
        self.globalScalings = globalScalings

        self.device = device
        self.conf_file = conf_file

        #sim info
        self.num_envs = num_envs
        self.scene = scene
        self.shadow_hands = robot
        self.dt = dt

        # other rendering tool
        self.visualize_gui = visualize_gui
        self.show_depth = show_depth
        self.render_tacto = render_tacto
        self.make_video = make_video
        self.render_sapien_camera = render_sapien_camera

        # init pytorch3d rendering
        self.init_tactile()

        if self.render_tacto == True:  # init pyrender rendering
            self.init_tacto()

        if self.render_sapien_camera:  #init isaac gym camera rendering
            self.init_sapien_camera()

        if self.make_video == True:  #init video
            self.writer = imageio.get_writer('../video/simulation3d.mp4',
                                             fps=60)

        self.index_dict()

    #==============================================================
    # create dictory for the index reference for digit
    #==============================================================

    def index_dict(self):

        self.digit_dict = {}

        for i, link in enumerate(self.robo_tactile_link):

            self.digit_dict[link.name] = i

    #==============================================================

    # obtain contact force
    #==============================================================

    def obtain_force(self, ):
        '''
        obtain the force of the robot link
        '''
        actor_set1 = set(self.obj_tactile_link)
        actor_set2 = set(self.robo_tactile_link)

        contact_forces = []
        contact_names = []
        force_index = []

        for contact in self.scene.get_contacts():
            contact_actors = {contact.actor0, contact.actor1}

            if len(actor_set1 & contact_actors) > 0 and len(
                    actor_set2 & contact_actors) > 0:

                impulse = np.array([point.impulse for point in contact.points])

                contact_force = np.linalg.norm(np.sum(impulse,
                                                      axis=0)) / (self.dt)

                if contact_force < 1:
                    continue

                if "tip" in contact.actor0.name and self.obj_tactile_link[
                        0].name in contact.actor1.name:

                    contact_names.append(contact.actor0)
                    force_index.append(self.digit_dict[contact.actor0.name])
                    contact_forces.append(
                        np.linalg.norm(np.sum(impulse, axis=0)) / (self.dt))

                elif "tip" in contact.actor1.name and self.obj_tactile_link[
                        0].name in contact.actor0.name:

                    contact_names.append(contact.actor1)
                    force_index.append(self.digit_dict[contact.actor1.name])
                    contact_forces.append(
                        np.linalg.norm(np.sum(impulse, axis=0)) / (self.dt))

        # if not contact_flag:
        #     contact_forces = None

        return contact_names, contact_forces, force_index,

    #==============================================================

    # update pose for all rigids
    #==============================================================

    def update_pose(self, contact_names):
        '''
        obtain the pos information for the related object according to the force info
        '''
        cameras_position = []
        cameras_quat = []
        obj_position = []
        obj_quat = []

        obj_trans = self.obj_tactile_link[0].get_pose().p
        obj_ori = self.obj_tactile_link[0].get_pose().q

        for contact_obj in contact_names:
            translation = contact_obj.get_pose().p
            quat = contact_obj.get_pose().q
            cameras_position.append(
                [translation[0], translation[1], translation[2]])
            cameras_quat.append([quat[0], quat[1], quat[2], quat[3]])

            obj_position.append([obj_trans[0], obj_trans[1], obj_trans[2]])
            obj_quat.append([obj_ori[0], obj_ori[1], obj_ori[2], obj_ori[3]])

        return cameras_position, cameras_quat, obj_position, obj_quat

    #==============================================================
    # numpy2torch
    #==============================================================
    def numpy2torch(self, data, repeat=False):

        data = torch.as_tensor(data, dtype=torch.float32, device=self.device)

        if repeat:
            data = torch.repeat_interleave(data, self.num_gel, dim=0)

        return data

    #==============================================================
    # obtain the render information
    #==============================================================
    def get_render_info(self):
        contact_names, contact_forces, force_index, = self.obtain_force()

        if not (not contact_forces):

            cameras_position, cameras_quat, obj_position, obj_quat = self.update_pose(
                contact_names)
            return cameras_position, cameras_quat, obj_position, obj_quat, list(
                contact_forces), force_index

        return None, None, None, None, None, None

    #==============================================================
    # init pytorch3d rendering
    #==============================================================
    def init_tactile(self):

        self.digits = Sensor(self.device,
                             self.conf_file,
                             self.num_envs,
                             self.obj_tactile_link,
                             self.robo_tactile_link,
                             self.num_cams,
                             self.urdf_files,
                             self.globalScalings,
                             self.scene,
                             background=self.background_img,
                             visualize_gui=True,
                             show_depth=True)

    #==============================================================
    # init pyrender rendering
    #==============================================================
    def init_tacto(self):
        '''
        init tactile simulation in the pyrender
        '''
        # background_img = cv2.imread("../conf/background.png")

        self.digits_tacto = Sensor_tacto(
            self.num_envs,  #number of envs in the simulation
            self.num_cams,  #num of cameras per robot
            width=120,
            height=160,
            background=None,
            config_path=get_digit_config_path(),
            visualize_gui=True,
            show_depth=True,
            zrange=0.002,
            cid=0,
        )
        self.renderer = Renderer_tacto(
            width=120,  # default for tacto
            height=160,  # default for tacto
            background=None,
            config_path=get_digit_config_path())

        body_urdf_path = self.urdf_files

        self.digits_tacto.add_body(body_urdf_path, self.globalScalings)

    #==============================================================
    # init isaac gym view rendering
    #==============================================================

    def init_sapien_camera(self):
        "init camera parameter in the isaac gym"

        near, far = 0.1, 100
        width, height = 640, 640
        camera_mount_actor = self.scene.create_actor_builder().build_kinematic(
        )
        self.camera = self.scene.add_mounted_camera(
            name="camera",
            actor=camera_mount_actor,
            pose=sapien.Pose(p=[1, 0, 0.5],
                             q=[0, 0, 0, 1]),  # relative to the mounted actor
            width=width,
            height=height,
            fovy=np.deg2rad(60),
            fovx=np.deg2rad(45),
            near=near,
            far=far,
        )
        self.writer_sapien = imageio.get_writer(
            '../video/simulation3d_sapien.mp4', fps=60)

    #==============================================================
    # update pyrender rendering
    #==============================================================
    def update_tacto(self, cameras_position, cameras_quat, obj_position,
                     obj_quat, contact_forces, force_index):

        # start = time.time()
        color, depth = self.digits_tacto.render(cameras_position, cameras_quat,
                                                obj_position, obj_quat,
                                                contact_forces, force_index)
        # end = time.time()
        # print("tacto time:", end - start)

        return color, depth

    #==============================================================

    # update render result for GUI if needed
    #==============================================================

    def updateGUI(self, colors, depths):
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

    #==============================================================
    # update pytorch3d rendering
    #==============================================================
    def update_tactile(self, cameras_position, cameras_quat, obj_position,
                       obj_quat, contact_forces, force_index):
        "update tactile simulation in the pyrender"
        # start = time.time()
        color, depth = self.digits.render(cameras_position, cameras_quat,
                                          obj_position, obj_quat,
                                          contact_forces,
                                          force_index)  # 12*64*64*3

        # end = time.time()
        # print("pytorch3d time:", end - start)

        if self.visualize_gui:
            self.digits.updateGUI(color, depth)

        return color, depth

    #==============================================================
    # update isaac gym rendering
    #==============================================================
    def update_sapien_render(self):
        '''
        render camera sensors
        '''

        self.camera.take_picture()
        rgba = self.camera.get_float_texture('Color')  # [H, W, 4]
        # An alias is also provided
        # rgba = camera.get_color_rgba()  # [H, W, 4]
        rgba_img = (rgba * 255).clip(0, 255).astype("uint8")
        rgba_pil = Image.fromarray(rgba_img)

        self.writer_sapien.append_data(rgba_img)

    #==============================================================
    # modify signgle depth channel to three channel
    #==============================================================
    def _depth_to_color(self, depth, zrange):
        "tactile depth modification to three channel images"

        gray = (np.clip(depth / zrange, 0, 1) * 255).astype(np.uint8)
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    #==============================================================
    # capture the video for rendering result
    #==============================================================
    def capture_video(self, tactile_color, tactile_depth, isaac_colors,
                      isaac_depth, tacto_color, tacto_depth):
        "create video for the simulation result"

        # concat pytorch3d rendeing image
        tactile_depth = np.array(tactile_depth)
        tactile_depth = self._depth_to_color(
            np.concatenate(tactile_depth, axis=1), 0.002)  # depth2rgb

        tactile_color = np.concatenate(tactile_color, axis=1)
        concat_image = np.concatenate(
            ((tactile_color * 255).astype(np.uint8), tactile_depth), axis=0)

        # # concat isaac gym rendeing image
        # if self.render_sapien_camera:
        #     isaac_depth = np.array(isaac_depth)
        #     isaac_depth = self._depth_to_color(
        #         np.concatenate(isaac_depth, axis=1) - np.min(isaac_depth),
        #         0.002)  # depth2rgb
        #     issac_color = np.concatenate(isaac_colors, axis=1)

        #     concat_image = np.concatenate((concat_image, issac_color), axis=0)
        #     concat_image = np.concatenate((concat_image, isaac_depth), axis=0)

        # concat pyrender rendeing image
        if self.render_tacto:

            tacto_depth = np.array(tacto_depth)
            tacto_depth = self._depth_to_color(
                np.concatenate(tacto_depth, axis=1) - np.min(tacto_depth),
                0.002)  # depth2rgb
            tacto_color = np.concatenate(tacto_color, axis=1)

            concat_image = np.concatenate((concat_image, (tacto_color)),
                                          axis=0)
            concat_image = np.concatenate((concat_image, tacto_depth), axis=0)

        self.writer.append_data(concat_image)

    #==============================================================
    # update rendering for per sim step
    #==============================================================
    def update_rendering(self):

        cameras_position, cameras_quat, obj_position, obj_quat, contact_forces, force_index = self.get_render_info(
        )

        tactile_color, tactile_depth = self.update_tactile(
            cameras_position, cameras_quat, obj_position, obj_quat,
            contact_forces, force_index)  # update pytorch3d rendering

        tacto_color, tacto_depth = None, None

        isaac_colors, isaac_depths = None, None

        if self.render_tacto:
            # start = time.time()
            tacto_color, tacto_depth = self.update_tacto(
                cameras_position, cameras_quat, obj_position, obj_quat,
                contact_forces, force_index)  # update tacto rendering
            # a = torch.as_tensor(tacto_color, device=self.device)

            # print('tacto time,', (time.time() - start) / self.num_envs / 4)

        if self.render_sapien_camera:
            self.update_sapien_render()  # update isaac gym rendering

        if self.make_video:
            self.capture_video(tactile_color.cpu().numpy(),
                               tactile_depth.cpu().numpy(), isaac_colors,
                               isaac_depths, tacto_color, tacto_depth)

        return tactile_color, tactile_depth
