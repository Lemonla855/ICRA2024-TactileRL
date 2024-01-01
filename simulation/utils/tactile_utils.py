from logging import captureWarnings
from hand_teleop.tactile_render.sensor_tacto import Sensor_tacto, get_digit_config_path
from hand_teleop.tactile_render.Render_tacto import Renderer_tacto
from hand_teleop.tactile_render.render3d import Render
from hand_teleop.utils import vision_BN3_1

from typing import List, Union
from urdfpy import URDF
from scipy.spatial.transform import Rotation as R
import sapien.core as sapien
import torch
import numpy as np
import cv2
import imageio
import torchvision.transforms.functional as F
# from torchvision.models.optical_flow import raft_large
# from torchvision.models.optical_flow import Raft_Large_Weights
# from torchvision.utils import flow_to_image
import torchvision
import sys
import time
from PIL import Image
import pdb
import trimesh
from scipy.spatial.transform import Rotation as R

from hand_teleop.utils.tactile_loader_utils import load_shapenet_object, load_ycb_object, load_egad_object, load_modelnet_object, load_spoon_object, load_partnet_actor_object
from hand_teleop.utils.tactile_loader_parnet_utils import load_partnet_object
from pathlib import Path
from hand_teleop.utils.egad_object_utils import EGAD_NAME, EGAD_KIND
from torchvision import transforms
import matplotlib.pyplot as plt
import math

import sys

sys.path.insert(0, '/media/lme/data/google_research/Grid-Keypoint-Learning')

from hand_teleop.utils.vision_BN3_1 import build_images_to_keypoints_net

from hand_teleop.utils.hyperparameters import get_config


#=======================================================
#   obtain tactile force
#=======================================================
def obtain_tactile_force(scene: sapien.Scene,
                         obj_tactile_link: List[sapien.Actor],
                         robo_tactile_link: List[sapien.Actor]) -> np.ndarray:
    '''
    args: scene: scene for the sapien
           obj_tactile_link : tactile link for manipulation objects
           robo_tactile_link : tactile link for the tip on the allegro hand
    return contact_forces
    '''
    actor_set2 = set(robo_tactile_link)
    dt = 0.004
    robot_link_names = [link.get_name() for link in robo_tactile_link]
    actor_set1 = set(obj_tactile_link)

    if not isinstance(obj_tactile_link[0], sapien.pysapien.Articulation):

        # actor_set1 = obj_tactile_link

        contact_forces = np.zeros((len(robo_tactile_link)))

        for contact in scene.get_contacts():
            contact_actors = {contact.actor0, contact.actor1}

            if len(actor_set1
                   & contact_actors) > 0 and len(actor_set2
                                                 & contact_actors) > 0:

                impulse = np.array([point.impulse for point in contact.points])

                contact_force = np.linalg.norm(np.sum(impulse, axis=0)) / (dt)

                if contact_force < 1e-2:  #contact forces is too smaller can be ingored
                    continue

                # only consider the contact between the tips and manipulated object
                if "tip" in contact.actor0.name and obj_tactile_link[
                        0].name in contact.actor1.name:

                    robot_link_index = robot_link_names.index(
                        contact.actor0.name)

                    contact_forces[robot_link_index] = (
                        np.linalg.norm(np.sum(impulse, axis=0)) / (dt))

                elif "tip" in contact.actor1.name and obj_tactile_link[
                        0].name in contact.actor0.name:

                    robot_link_index = robot_link_names.index(
                        contact.actor1.name)
                    contact_forces[robot_link_index] = (
                        np.linalg.norm(np.sum(impulse, axis=0)) / (dt))
    else:

        actor_set1 = set(obj_tactile_link[0].get_links())
        contact_forces = np.zeros((len(robo_tactile_link)))

        id_list = []
        for link in actor_set1:
            id_list.append(link.id)  # partnet object link id

        for contact in scene.get_contacts():
            contact_actors = {contact.actor0, contact.actor1}

            if len(actor_set1
                   & contact_actors) > 0 and len(actor_set2
                                                 & contact_actors) > 0:

                if (
                    (contact.actor0.id not in id_list)
                        and (contact.actor1.id not in id_list)
                ):  # the link belong to robot with the same name, there is not contact with the manipulated object
                    continue

                impulse = np.array([point.impulse for point in contact.points])

                contact_force = np.linalg.norm(np.sum(impulse, axis=0)) / (dt)

                if contact_force < 1e-2:  #contact forces is too smaller can be ingored
                    continue

                # only consider the contact between the tips and manipulated object
                if "tip" in contact.actor0.name and obj_tactile_link[
                        0].name in contact.actor1.name:

                    robot_link_index = robot_link_names.index(
                        contact.actor0.name)

                    contact_forces[robot_link_index] = (
                        np.linalg.norm(np.sum(impulse, axis=0)) / (dt))

                elif "tip" in contact.actor1.name and obj_tactile_link[
                        0].name in contact.actor0.name:

                    robot_link_index = robot_link_names.index(
                        contact.actor1.name)
                    contact_forces[robot_link_index] = (
                        np.linalg.norm(np.sum(impulse, axis=0)) / (dt))

    return contact_forces


#=======================================================
#   obtain tactile position
#=======================================================
def obtain_tactile_postion(obj_tactile_link: List[sapien.Actor],
                           robo_tactile_link: List[sapien.Actor],
                           render_link=None) -> np.ndarray:
    '''
    args : obj_tactile_link,robo_tactile_link
    return : robo_pose and obj_pose 
    (x,y,z,w,qx,qy,qz) robot, obj
    '''
    robo_pose = []
    obj_pose = []

    obj_pos = obj_tactile_link.get_pose().p
    obj_quat = obj_tactile_link.get_pose().q

    for link in robo_tactile_link:

        link_pos = link.get_pose().p
        link_quat = link.get_pose().q

        robo_pose.append((list(np.concatenate((link_pos, link_quat), axis=0))))

        if not isinstance(obj_tactile_link, sapien.pysapien.Articulation):
            obj_pose.append((list(np.concatenate((obj_pos, obj_quat),
                                                 axis=0))))
        else:
            obj_links_pose = [
                np.concatenate((link.get_pose().p, link.get_pose().q), axis=0)
                for link in render_link
            ]  # N*7

            obj_pose.append(np.concatenate(np.array(obj_links_pose), axis=0))

    return np.concatenate((robo_pose, obj_pose), axis=1)  #(N+1)*7

    # if not isinstance(obj_tactile_link[0], sapien.pysapien.Articulation):
    #     return np.concatenate((robo_pose, obj_pose), axis=1)
    # else:
    #     return np.concatenate((robo_pose,  obj_pose), axis=1)


def get_conf_root_dir():
    current_dir = Path(__file__).parent
    tactile_conf_dir = current_dir.parent.parent / "conf"
    return tactile_conf_dir.resolve()


tactile_conf_dir = get_conf_root_dir()


def load_object(object_category, object_name, novel=False):
    any = False

    if object_category.lower() == "ycb":
        if object_name == "any_eval" or object_name == "any_train":
            any = True
        obj_meshes, obj_names = load_ycb_object(object_name)

    elif object_category.lower() == "egad":

        if object_name == "any_eval":
            names = EGAD_NAME["eval"]
            obj_meshes, obj_names = load_egad_object(names)

        elif object_name == "any_train":
            names = EGAD_NAME["eval"]
            obj_meshes, obj_names = load_egad_object(names)
            any = True

        elif len(object_name) == 1:  #load a series of object
            names = EGAD_KIND[object_name]
            obj_meshes, obj_names = load_egad_object(names)
            any = True

        else:  #load a single object

            obj_meshes, obj_names = load_egad_object([object_name])

    elif object_category.lower() in [
            "pen", "usb", "knife", "bottle", "breaking", "breaking_bottle",
            "box", "bucket"
    ]:

        obj_meshes, obj_names = load_partnet_actor_object(object_category,
                                                          object_name,
                                                          novel=novel)
        if object_name == "any_train":
            obj_meshes, obj_names = load_partnet_actor_object(object_category,
                                                              object_name,
                                                              novel=novel)
            any = True
        any = False

    elif object_category.isnumeric():
        any = True

        if object_name == "any_eval":
            obj_meshes, obj_names = load_shapenet_object(object_category,
                                                         key="eval")

        if object_name == "any_train":
            obj_meshes, obj_names = load_shapenet_object(object_category,
                                                         key="train")
        else:
            obj_meshes, obj_names = load_shapenet_object(object_category,
                                                         object_name,
                                                         key="train")
            any = False

    elif object_category.lower() == "modelnet":
        obj_meshes, obj_names = load_modelnet_object(object_name)
        if object_name == "any_train":
            obj_meshes, obj_names = load_modelnet_object(object_name, )
            any = True
        any = False

    elif object_category.lower() in ["spoon", "kitchen"]:
        obj_meshes, obj_names = load_spoon_object(object_category, object_name)
        if object_name == "any_train":
            obj_meshes, obj_names = load_spoon_object(object_category,
                                                      object_name)
            any = True
        any = False

    elif "any" in object_category.lower():  #["pen",spoon]

        partnet_obj_meshes, partnet_obj_names = load_partnet_actor_object(
            "pen", object_name)

        spoon_obj_meshes, spoon_obj_names = load_spoon_object(
            "spoon", object_name)
        obj_meshes = partnet_obj_meshes + spoon_obj_meshes
        obj_names = partnet_obj_names + spoon_obj_names

        any = True

    elif "all" in object_category.lower():  #["pen",spoon]

        partnet_obj_meshes, partnet_obj_names = load_partnet_actor_object(
            "pen", object_name, novel=novel)

        box_obj_meshes, box_obj_names = load_partnet_actor_object("box",
                                                                  object_name,
                                                                  novel=novel)

        bottle_obj_meshes, bottle_obj_names = load_partnet_actor_object(
            "bottle", object_name, novel=novel)

        obj_meshes = partnet_obj_meshes + box_obj_meshes + bottle_obj_meshes
        obj_names = partnet_obj_names + box_obj_names + bottle_obj_names

        any = True

    else:
        if "any" in object_name:
            any = True
        else:
            any = False

        obj_meshes, obj_names = load_partnet_object(object_category,
                                                    [object_name])

    return obj_meshes, obj_names, any


#=====================tactile===========================
def init_tactile(obj_name, device, **tactile_kwargs):

    config = str(tactile_conf_dir / "rendering.yaml")

    bg_img = cv2.imread(str(tactile_conf_dir / "background.png"))

    if "object_category" in tactile_kwargs.keys():
        obj_catergory = tactile_kwargs["object_category"]
    else:  # not specify object_category will be regarded as YCB object
        obj_catergory = "YCB"

    if "novel" in tactile_kwargs.keys():
        novel = tactile_kwargs["novel"]
    else:
        novel = False

    obj_meshes, obj_names, any = load_object(obj_catergory, obj_name, novel)

    tactile = Tactile(
        device=device,
        conf_file=config,
        num_envs=1,
        num_cameras=tactile_kwargs['num_cameras']
        if "num_cameras" in tactile_kwargs.keys() else 4,
        obj_catergory=obj_catergory,
        obj_names=obj_names,
        obj_meshes=obj_meshes,
        background=bg_img,
        use_rgb=tactile_kwargs['use_rgb']
        if "use_rgb" in tactile_kwargs.keys() else False,
        use_depth=tactile_kwargs['use_depth']
        if "use_depth" in tactile_kwargs.keys() else False,
        visualize_gui=tactile_kwargs['visualize_gui']
        if "visualize_gui" in tactile_kwargs.keys() else False,
        show_depth=tactile_kwargs['show_depth']
        if "show_depth" in tactile_kwargs.keys() else False,
        render_tacto=tactile_kwargs['render_tacto']
        if "render_tacto" in tactile_kwargs.keys() else False,
        render_sapien_camera=tactile_kwargs['render_sapien_camera']
        if "render_sapien_camera" in tactile_kwargs.keys() else False,
        make_video=tactile_kwargs['make_video']
        if "make_video" in tactile_kwargs.keys() else False,
        scene=tactile_kwargs['scene']
        if "scene" in tactile_kwargs.keys() else False,
        any_train=any,
        use_mask=tactile_kwargs["use_mask"]
        if "use_mask" in tactile_kwargs.keys() else False,
        add_noise=tactile_kwargs["add_noise"]
        if "add_noise" in tactile_kwargs.keys() else False,
        use_diff=tactile_kwargs["use_diff"]
        if "use_diff" in tactile_kwargs.keys() else False,
        image_size=tactile_kwargs["image_size"]
        if "image_size" in tactile_kwargs.keys() else 64,
        optical_flow=tactile_kwargs["optical_flow"]
        if "optical_flow" in tactile_kwargs.keys() else False,
        keypoints=tactile_kwargs["keypoints"]
        if "keypoints" in tactile_kwargs.keys() else False,
        task_index=tactile_kwargs["task_index"]
        if "task_index" in tactile_kwargs.keys() else False,
        crop=tactile_kwargs["crop"]
        if "crop" in tactile_kwargs.keys() else False,
        show_gui=tactile_kwargs["show_gui"]
        if "show_gui" in tactile_kwargs.keys() else False,
        random=tactile_kwargs["random"]
        if "random" in tactile_kwargs.keys() else False,
    )
    return tactile

    #=====================tactile===========================


#=====================tactile===========================
#render tactile image from tactile images informations
#=====================tactile===========================


def state2tactile(tactile,
                  tactile_state,
                  reset: bool = False,
                  eval: bool = False):
    '''
    To speed up the rendering ,the rendering batch size is better to be more than 16. 
    Since the collecting experience data is used multiprocessing, the GPU rendering 
    can not be implemented during the multiprocessing. Here will rendering  the images
    obtained from multiprocessing in a batch.
    '''

    tactile_state = torch.as_tensor(
        tactile_state,
        dtype=torch.float32,
    )

    tactile_state = tactile_state.view(-1, tactile_state.shape[-1])

    tacile_image = tactile.render(tactile_state, reset, eval)

    # obs_dict["tactile_image"] = tacile_image

    return (tacile_image.cpu().numpy() * 255)  #.astype(np.uint8)


#=====================tactile===========================
#=======================================================


class Tactile:

    def __init__(self,
                 device,
                 conf_file,
                 num_envs,
                 num_cameras,
                 obj_catergory,
                 obj_names,
                 obj_meshes,
                 background,
                 use_rgb=True,
                 use_depth=False,
                 visualize_gui=False,
                 show_depth=False,
                 render_tacto=False,
                 render_sapien_camera=False,
                 make_video=False,
                 scene=None,
                 any_train=False,
                 use_mask=False,
                 add_noise=False,
                 use_diff=False,
                 image_size=64,
                 optical_flow=False,
                 keypoints=False,
                 crop=False,
                 task_index=0,
                 show_gui=False,
                 random=False):

        self.device = device
        self.conf_file = conf_file

        self.num_envs = num_envs
        self.num_cameras = num_cameras
        self.scene = scene

        self.objects_mesh = []
        self.obj_names = obj_names
        self.any_train = any_train
        self.optical_flow = optical_flow  # use optical flow or not
        self.flow_imgs = None
        self.keypoints = keypoints
        self.obj_catergory = obj_catergory
        self.random_domain = random  #random the image domain

        #init visulize
        self.show_gui = show_gui
        self.visualize_gui = visualize_gui
        self.show_depth = show_depth
        self.render_tacto = render_tacto
        self.render_sapien_camera = render_sapien_camera
        self.render_sapien_camera = render_sapien_camera
        self.make_video = make_video
        self.use_mask = use_mask  #use mask or not
        self.use_depth = use_depth  #use depth or not
        self.use_rgb = use_rgb  #use rgb image or nor
        self.use_diff = use_diff  #use rgb diff image
        self.crop = crop  #crop image or not

        assert not (self.use_diff and self.optical_flow)  # avoid conflict

        self.add_noise = add_noise  #add noise for the images
        self.image_size = image_size  # the size of image

        self.init_object(obj_meshes)

        self.renderer = Render(device,
                               obj_catergory,
                               self.num_envs,
                               background,
                               conf_file,
                               self.objects_mesh,
                               num_cameras,
                               self.obj_names,
                               self.image_size,
                               task_index=task_index)  #init renderer

        # self.init_render_static()  # render static image in advance
        self.rgb, self.depth = self.renderer.render_static()

        if self.render_tacto == True:  # init pyrender rendering
            self.init_tacto()

        if self.render_sapien_camera:  #init isaac gym camera rendering
            self.init_sapien_camera()

        if self.make_video == True:  #init video
            self.writer = imageio.get_writer('./video/simulation3d.mp4',
                                             fps=60)

        self.obj_index = 0

        if self.add_noise:
            self.GaussianBlur = transforms.GaussianBlur(11, sigma=(0.1, 0.1))
            self.sigma = 0.1

        if self.optical_flow:
            self.init_opticalflow()

        if self.keypoints:
            self.pre_keypoints_list = None
            cfg = get_config()
            cfg.num_keypoints = 4

            self.build_images_to_keypoints_net = build_images_to_keypoints_net(
                cfg, [3, self.image_size, self.image_size]).cuda()
            # keypoints_model = torch.load(
            #     "/media/lme/data/google_research/Grid-Keypoint-Learning/logs/1211_keypoints/struc-tactile4/model_989.pth/"
            # )

            state_dict = torch.load(
                "/media/lme/data/google_research/Grid-Keypoint-Learning/logs/1211_keypoints/struc-tactile4/model_989.pth"
            )['build_images_to_keypoints_net']
            self.build_images_to_keypoints_net.load_state_dict(state_dict)
            # self.build_images_to_keypoints_net.eval()

    #==============================================================
    # preprocess optical flow images
    #==============================================================
    def preprocess(self, img1_batch, img2_batch):

        # weights = Raft_Large_Weights.DEFAULT
        # transforms = weights.transforms()
        img1_batch = F.resize(img1_batch, size=[128, 128])
        img2_batch = F.resize(img2_batch, size=[128, 128])

        return self.optical_transforms(img1_batch, img2_batch)

    #==============================================================
    # init optical flow
    #==============================================================
    def init_opticalflow(self):

        model = raft_large(weights=Raft_Large_Weights.DEFAULT,
                           progress=False).to(self.device)

        self.optical_model = model.eval()
        self.pre_image_batch = None
        weights = Raft_Large_Weights.DEFAULT
        self.optical_transforms = weights.transforms()

        plt.style.use('seaborn')
        plt.axis("off")

        self.fig_flowdir = plt.figure()
        self.ax_flowdir = self.fig_flowdir.add_subplot()
        # self.pre_image_batch = torch.transpose(self.pre_image_batch, 1, 3)
        # self.pre_image_batch = torch.transpose(self.pre_image_batch, 2,
        #                                        3)  # channel first

    #==============================================================
    # init pyrender rendering
    #==============================================================
    def init_tacto(self) -> None:
        '''
        init tactile simulation in the pyrender
        '''
        # background_img = cv2.imread("../conf/background.png")

        self.digits_tacto = Sensor_tacto(
            self.num_envs,  #number of envs in the simulation
            self.num_cameras,  #num of cameras per robot
            width=120,
            height=160,
            background=None,
            config_path=get_digit_config_path(),
            visualize_gui=True,
            show_depth=True,
            zrange=0.002,
            cid=0,
        )
        self.renderer_tacto = Renderer_tacto(
            width=120,  # default for tacto
            height=160,  # default for tacto
            background=None,
            config_path=get_digit_config_path())

    #==============================================================
    # render static images  for no contact
    #==============================================================
    def render_static(self,
                      num_images: int = 4
                      ) -> Union[torch.Tensor, torch.Tensor]:
        '''
        render the images for no contact scene
        '''

        rgb0 = torch.repeat_interleave(self.rgb[:, :, :, :3],
                                       num_images,
                                       dim=0)

        depth0 = torch.repeat_interleave(torch.unsqueeze(self.depth * 0.0,
                                                         dim=0),
                                         num_images,
                                         dim=0)

        return rgb0, depth0

    #==============================================================
    #add object
    #==============================================================

    def init_object(self, obj_meshes) -> None:
        '''
        pre load the mesh for rendering
        '''

        if len(obj_meshes) == 1:
            self.obj_index = 0
        else:
            self.obj_index = None

        if len(obj_meshes[0]) > 1:  #multi links

            for index, mesh in enumerate(obj_meshes):
                faces = torch.as_tensor(mesh[1],
                                        dtype=torch.float32,
                                        device=self.device)

                temp_mesh = []

                for m in mesh[0]:
                    m = torch.as_tensor(m,
                                        dtype=torch.float32,
                                        device=self.device)
                    temp_mesh.append(m)

                self.objects_mesh.append([temp_mesh, faces])
                # ax = plt.axes(projection='3d')
                # # points_pixel = pixel_coords[0].cpu().view(-1, 1, 3)
                # # ax.scatter(points_pixel[:, :, 0].reshape(-1),
                # #            points_pixel[:, :, 1].reshape(-1),
                # #            points_pixel[:, :, 2].reshape(-1),
                # #            c=points_pixel[:, :, 0].reshape(-1))
                # points_pixel = self.objects_mesh[index][0][0].cpu().view(
                #     -1, 1, 3)
                # ax.scatter(points_pixel[:, :, 0].reshape(-1),
                #            points_pixel[:, :, 1].reshape(-1),
                #            points_pixel[:, :, 2].reshape(-1),
                #            c='r')
                # # points_pixel = gel_mesh[-1].cpu().view(-1, 1, 3)
                # # ax.scatter(points_pixel[:, :, 0].reshape(-1),
                # #            points_pixel[:, :, 1].reshape(-1),
                # #            points_pixel[:, :, 2].reshape(-1),
                # #            c='b')
                # plt.show()

        # else:
        # for mesh in obj_meshes:
        #     vertices = torch.as_tensor(mesh[0],
        #                                 dtype=torch.float32,
        #                                 device=self.device)
        #     faces = torch.as_tensor(mesh[1],
        #                             dtype=torch.float32,
        #                             device=self.device)

        #     self.objects_mesh.append([vertices, faces])

    #==============================================================
    # numpy2torch
    #==============================================================

    def numpy2torch(self, data: torch.Tensor, repeat=False) -> torch.Tensor:
        '''
        transfer numpy to torch
        '''

        data = torch.as_tensor(data, dtype=torch.float32, device=self.device)

        if repeat:
            data = torch.repeat_interleave(data, self.num_cameras, dim=0)

        return data

    #==============================================================
    # generate render information
    #==============================================================
    def generate_render_info(
        self,
        cameras_position: torch.Tensor,
        cameras_quat: torch.Tensor,
        obj_position: torch.Tensor,
        obj_quat: torch.Tensor,
        contact_forces: torch.Tensor,
        obj_indexes: torch.Tensor,
        obj_scales: torch.Tensor,
    ) -> List[torch.Tensor]:
        '''
        generate render info,
        To reduce the numbder of rendering images, just render the tactile image for 
        the digit which has contact with the manipulated object and just render the 
        images with nonzero force
        args:robot tactile links and manipulated object position and orientation
        return: the reduced version of args paras.
        '''
        if isinstance(
                contact_forces,
                torch.Tensor):  #tactile data for pytorch3d rendering on GPU

            force_index, y_index = torch.where(contact_forces > 1e-2)

        elif isinstance(
                contact_forces,
                np.ndarray):  #tactile data for pyrender rendering on CPU
            force_index, y_index = np.where(contact_forces > 1e-2)

        # decrease the related information to the sets with nonzero contact forces
        decreased_cameras_position = cameras_position[force_index]
        decreased_cameras_quat = cameras_quat[force_index]
        decreased_obj_position = obj_position[force_index]
        decreased_obj_quat = obj_quat[force_index]

        decreased_contact_forces = contact_forces[force_index]

        obj_indexes = torch.as_tensor(obj_indexes[force_index].reshape(-1),
                                      dtype=torch.int32)

        obj_scales = torch.as_tensor(obj_scales[force_index].reshape(-1, 3))

        return decreased_cameras_position, decreased_cameras_quat, decreased_obj_position, decreased_obj_quat, decreased_contact_forces, force_index, obj_indexes, obj_scales

    def noise_imgs(self, imgs):
        assert isinstance(imgs, torch.Tensor)
        imgs = self.GaussianBlur(imgs)  #blur the images

        out = imgs + self.sigma * torch.randn_like(imgs) / 255

        return out

    #==============================================================
    # transpose images according batch size and channel first
    #==============================================================
    def transpose_imgs(self, output_color: torch.Tensor) -> np.ndarray:
        '''
        transpose images to channel first and stack as (B,channels,W,H)
        '''

        if isinstance(output_color, torch.Tensor):  #GPU for pytorch3d render
            #read dimension firt
            dim = output_color.shape[3]

            # if self.use_mask:  #mask dimension(N,4,64,64,4)
            #     imgs = output_color.view(-1, 4, output_color.shape[1],
            #                              output_color.shape[1], 4)
            # else:  #color dimension(N,4,64,64,3)
            #     imgs = output_color.view(-1, 4, output_color.shape[1],
            #                              output_color.shape[1], 3)
            imgs = torch.transpose(output_color, 1, 3)
            imgs = torch.transpose(imgs, 2, 3)
            import torchvision

            resize_images = (torchvision.transforms.Resize(
                size=(self.image_size, self.image_size))(imgs))

            if self.random_domain:
                if self.use_rgb:
                    transform = transforms.Compose([
                        transforms.ColorJitter(
                            brightness=(0.5, 1.0),
                            contrast=(0., 1.0),
                            # saturation=(0.1, 0.8),
                            hue=(-0.5, 0.5)),  # Adjust color
                    ])
                    resize_images = transform(resize_images)
                else:

                    transform = transforms.Compose([
                        transforms.RandomAffine(
                            degrees=0, translate=(0.0, 0.0), scale=(
                                0.5,
                                1.0)),  # Apply random affine transformations
                        # transforms.ToTensor(),  # Convert the image to a PyTorch tensor
                        # transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        #                      std=[0.229, 0.224, 0.225])  # Normalize the image
                        transforms.RandomErasing(scale=(0.03, 0.3)),
                    ])
                    resize_images = transform(resize_images)

            # imgs = torch.zeros(
            #     (int(resize_images.shape[0] / self.num_cameras),
            #      self.num_cameras * dim, self.image_size, self.image_size))

            # for i in range(imgs.shape[0]):
            #     for j in range(self.num_cameras):
            #         imgs[i, j * 3:(j + 1) *
            #              3] = resize_images[self.num_cameras * i + j]

            imgs = resize_images.reshape(-1, self.num_cameras * dim,
                                         self.image_size, self.image_size)

            # if self.use_rgb:
            imgs[:, dim:, :, :] = torch.flip(imgs[:, dim:, :, :], dims=[3])
            # else:
            #     imgs[:, dim:, :, :] = torch.flip(imgs[:, dim:, :, :], dims=[2])
            input_tensor = imgs.transpose(1, 0).cpu()

            # if self.crop:

            #     for i in range(4):
            #         crop_size = 16
            #         transform = transforms.RandomCrop(size=crop_size, fill=0)
            #         # Create a random crop
            #         i, j, h, w = transform.get_params(input_tensor,
            #                                           (crop_size, crop_size))
            #         # Create a mask of ones for the crop region and zeros elsewhere
            #         mask = torch.ones(input_tensor.shape[0], 1, 64, 64)
            #         mask[:, :, i:i + h, j:j + w] = 0

            #         # Replace the crop region with zeros
            #         input_tensor = input_tensor * (mask)

            imgs = input_tensor.transpose(1, 0)

            # imgs = torch.cat([resize_images[0], resize_images[1]])
            # if self.use_mask:
            #     imgs = imgs.reshape(-1, 16, output_color.shape[1],
            #                         output_color.shape[1])
            # else:
            #     imgs = imgs.reshape(-1, 12, output_color.shape[1],
            #                         output_color.shape[1])

        elif isinstance(output_color, np.ndarray):  #CPU for pyrender
            imgs = output_color.reshape(-1, self.num_cameras,
                                        output_color.shape[1],
                                        output_color.shape[1], 3)
            imgs = np.transpose(imgs, (0, 1, 4, 2, 3))
            imgs = np.array(imgs).reshape(-1, self.num_cameras * dim,
                                          output_color.shape[1],
                                          output_color.shape[1])

        if self.add_noise:
            imgs = self.noise_imgs(imgs)

        return imgs

    #==============================================================
    # split the tatile state into cameras and obj transformation
    #==============================================================
    def split_state(
        self, tactile_states: torch.Tensor
    ) -> Union[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
               torch.Tensor]:
        '''
        split the tactile state
        return: render info for position and orientation and contact force
        '''

        contact_forces = tactile_states[:, 14].reshape(
            -1, 1)  # the contact force of the obj

        cameras_position = tactile_states[:, :3].reshape(
            -1, 3)  # postion for camera
        cameras_quat = tactile_states[:, 3:7].reshape(-1, 4)  # quat for camera

        obj_state = tactile_states[:, 7:14].reshape(tactile_states.shape[0],
                                                    -1, 7)

        obj_position = obj_state[:, :, 0:3]  # num gel * num link *7
        obj_quat = obj_state[:, :, 3:]
        # obj_position = tactile_states[:, 7:10].reshape(-1,
        #                                                3)  # postion for object
        # obj_quat = tactile_states[:, 10:-2].reshape(-1, 4)  # quat for objects
        obj_indexes = tactile_states[:, -4].reshape(-1,
                                                    1)  # the index of the obj
        obj_scales = tactile_states[:, -3:].reshape(-1,
                                                    3)  # the index of the obj

        #render with less information
        decreased_cameras_position, decreased_cameras_quat, decreased_obj_position, decreased_obj_quat, decreased_contact_forces, force_index, obj_indexes, obj_scales = self.generate_render_info(
            cameras_position, cameras_quat, obj_position, obj_quat,
            contact_forces, obj_indexes, obj_scales)

        return decreased_cameras_position, decreased_cameras_quat, decreased_obj_position, decreased_obj_quat, decreased_contact_forces, force_index, obj_indexes, obj_scales

    #==============================================================
    # visual optical flow direction
    #==============================================================
    def visual_flow(self):

        x = np.arange(0, 128, 1)
        y = np.arange(0, 128 * 4, 1)
        xx, yy = np.meshgrid(x, y, sparse=True)  # 为一维的矩阵
        xx1, yy1 = np.meshgrid(x, y)

        flow_imgs = np.transpose(self.predicted_flows.cpu().numpy(),
                                 (0, 2, 3, 1))

        flow_imgs = (np.concatenate(flow_imgs, axis=1))

        self.ax_flowdir.quiver(xx1, yy1, flow_imgs[:, :, 0], flow_imgs[:, :,
                                                                       1])
        plt.pause(0.1)

        # fig.canvas.draw()

        # # convert canvas to image
        # img = (np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8,
        #                      sep=''))
        # img = (img.reshape(fig.canvas.get_width_height()[::-1] + (3, )) /
        #        255).astype(np.float32)

        # img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        # img = cv2.resize(img, (128 * 4, 128))

        # return img

    def draw_arrow(self):
        # Green color in BGR
        color = (0, 1, 0)
        thickness = 2
        font = cv2.FONT_HERSHEY_SIMPLEX
        arrow_images = []
        if self.pre_keypoints_list is not None:

            pre_mean = np.mean(self.pre_keypoints_list,
                               axis=1).astype(np.uint8)
            now_mean = np.mean(self.keypoinsts_list, axis=1).astype(np.uint8)

            for i in range(self.num_cameras):
                image = np.ones((self.image_size, self.image_size, 3))

                # for j in range(len(self.keypoinsts_list[i])):
                #     # End coordinate
                #     end_point = (self.keypoinsts_list[i][j][0],
                #                  self.keypoinsts_list[i][j][1])
                #     start_point = (self.pre_keypoints_list[i][j][0],
                #                    self.pre_keypoints_list[i][j][1])
                #     # Line thickness of 9 px
                #     thickness = 2

                #     # Using cv2.arrowedLine() method
                #     # Draw a diagonal green arrow line
                #     # with thickness of 9 px
                #     image = cv2.arrowedLine(image, start_point, end_point,
                #                             color, thickness)

                start_point = (pre_mean[i][0], pre_mean[i][1])
                end_point = (now_mean[i][0], now_mean[i][1])

                # image = cv2.arrowedLine(image, start_point, end_point, color,
                #                         thickness)
                image = cv2.putText(
                    image,
                    str((np.array(now_mean).astype(np.float32) -
                         np.array(pre_mean).astype(np.float32))[i, 0]),
                    (32, 32), font, 0.5, (0, 255, 0), 2, cv2.LINE_AA)

                arrow_images.append(image)

            # print((np.array(now_mean).astype(np.float32) -
            #        np.array(pre_mean).astype(np.float32))[:, 1])

            self.pre_keypoints_list = self.keypoinsts_list
            return arrow_images
        self.pre_keypoints_list = self.keypoinsts_list

        for i in range(self.num_cameras):
            image = np.ones((self.image_size, self.image_size, 3))
            arrow_images.append(image)
        return arrow_images

    #==============================================================
    # update render result for GUI if needed
    #==============================================================
    def updateGUI(self, colors: np.ndarray, depths: np.ndarray) -> None:
        """
        Update images for visualization
        """
        if not self.visualize_gui:
            return

        color = np.concatenate((colors).cpu().numpy(),
                               axis=1)  #.astype(np.uint8)

        color_n_depth = [color]

        if self.show_depth:

            # depth = self.depth_to_color(
            #     np.concatenate(depths.cpu().numpy(), axis=1), 0.002)
            imgs = colors - self.flip_static_color

            diff = imgs + 0.5
            diff[diff < 0.5] = (diff[diff < 0.5] - 0.5) * 0.7 + 0.5
            diff_imgs = torch.mean(torch.abs(diff - 0.5), axis=-1)[:, :, :,
                                                                   None]

            mask = torch.as_tensor((torch.abs(diff_imgs) > 0.06),
                                   dtype=torch.float32)

            diff_imgs /= torch.clip(
                torch.max(diff_imgs.view(diff_imgs.shape[0], -1),
                          dim=1).values, 0.01, 1)[:, None, None, None]
            # diff_imgs = ((torch.abs(diff_imgs) > 0.15))
            mask = torch.cat([mask, mask, mask], axis=-1)
            mask = np.concatenate(mask.cpu().numpy(), axis=1)

            diff_imgs = torch.cat([diff_imgs, diff_imgs, diff_imgs], axis=-1)
            diff_imgs = np.concatenate(diff_imgs.cpu().numpy(), axis=1)

            # color_n_depth.append(np.array(diff_imgs*255).astype(np.uint8))
            # color_n_depth.append(np.array(mask*255).astype(np.uint8))

            color_n_depth.append(diff_imgs)
            color_n_depth.append(mask)

        if self.flow_imgs is not None:

            flow_imgs = F.resize(self.flow_imgs, size=[64, 64]).cpu().numpy()
            flow_imgs = np.transpose(flow_imgs, (0, 2, 3, 1))
            flow_imgs = (np.concatenate(flow_imgs, axis=1) / 255).astype(
                np.float32)
            color_n_depth.append(flow_imgs)

        if self.use_diff:
            diff_color = (np.concatenate(
                (self.diff_imgs).cpu().numpy(), axis=1) * 255).astype(np.uint8)
            # plt.imshow(diff_color)
            # plt.show()
            color_n_depth.append(diff_color)

        if self.keypoints:

            heatmap_color = self.depth_to_color(
                np.concatenate((self.keypoints_row), axis=1), 0.002)
            color_n_depth.append(heatmap_color)
            arrow_images = self.draw_arrow()

            if arrow_images is not None:
                arrow_images = np.concatenate((arrow_images),
                                              axis=1)  #.astype(np.uint8)

                color_n_depth.append(
                    np.array(arrow_images * 255).astype(np.uint8))

        color_n_depth = np.concatenate(color_n_depth, axis=0)
        # color_n_depth = cv2.resize(color_n_depth, (16, 16))
        if self.show_gui:
            cv2.imshow("color and depth",
                       cv2.cvtColor(color_n_depth, cv2.COLOR_RGB2BGR))

            cv2.waitKey(1)

        return color_n_depth

    def render_opticalflow_image(self, image_batch, pre_image_batch):

        imgs = torch.transpose(image_batch, 1, 3)
        imgs = torch.transpose(imgs, 2, 3)  # channel first

        pre_image_batch = torch.transpose(pre_image_batch, 1, 3)
        pre_image_batch = torch.transpose(pre_image_batch, 2, 3)
        flow_imgs = []

        for i in range(math.ceil(image_batch.shape[0] / 10)):

            img1_batch = torch.as_tensor((imgs[i * 10:i * 10 + 10]))
            img2_batch = torch.as_tensor((pre_image_batch[i * 10:i * 10 + 10]))
            img1_batch, img2_batch = self.preprocess(img1_batch, img2_batch)

            with torch.no_grad():
                list_of_flows = self.optical_model(img1_batch, img2_batch)
            self.predicted_flows = list_of_flows[-1]
            flow_imgs.append(flow_to_image(self.predicted_flows))

        self.flow_imgs = torch.cat(flow_imgs)

        return self.flow_imgs

    def depth_to_color(self, depth: np.ndarray, zrange: float = 0.002):
        "tactile depth modification to three channel images"

        gray = (np.clip(depth / zrange, 0, 1) * 255).astype(np.uint8)
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    #==============================================================
    # visulize the result in the evaluation process
    #==============================================================
    def visulize(self, tactile_state: torch.Tensor) -> None:
        '''
        visulize the result just for evaluation process
        '''

        self.update_obj_index(tactile_state[:, -1])
        tactile_state = torch.as_tensor(tactile_state,
                                        dtype=torch.float32,
                                        device=self.device)

        tactile_color, tactile_depth = self.render_static(len(tactile_state))

        contact_forces = tactile_state[:, 14].reshape(-1, 1)

        # if torch.max(contact_forces) < 1e-2:

        #     tactile_color, tactile_depth = tactile_color, tactile_depth

        # else:

        # decreased_cameras_position, decreased_cameras_quat, decreased_obj_position, decreased_obj_quat, decreased_contact_forces, force_index, obj_indexes, obj_scales = self.split_state(
        #     tactile_state)

        # decreased_tactile_color,decreased_tactile_depth=self.visulize_tactile(decreased_cameras_position, decreased_cameras_quat, decreased_obj_position,\
        # decreased_obj_quat, decreased_contact_forces,obj_indexes,obj_scales)

        # tactile_color[force_index] = decreased_tactile_color
        # tactile_depth[force_index] = decreased_tactile_depth
        self.render(tactile_state)

        tactile_color = self.visual_color

        # self.visual_depth[1] = torch.flip(self.visual_depth[1], [1])

        tactile_depth = self.visual_depth

        if self.keypoints:

            keypoints_img = torch.transpose(tactile_color, 1, 3)
            keypoints_img = torch.transpose(keypoints_img, 2, 3)

            with torch.no_grad():
                keypoints_index, self.heatmap = self.build_images_to_keypoints_net(
                    keypoints_img - 0.5)
                self.heatmap = (torchvision.transforms.Resize(
                    size=(self.image_size,
                          self.image_size))(self.heatmap)).cpu().numpy()
                self.heatmap = np.transpose(self.heatmap, (0, 2, 3, 1))

                radius = 1

            self.keypoints_row = []
            self.keypoinsts_list = []
            # for t in range(len(keypoints_img)):
            #     im_key = torch.zeros((self.image_size, self.image_size))
            # im_key = torch.zeros(
            #     (len(keypoints_index), self.image_size, self.image_size))
            # pdb.set_trace()
            keypoints_index = keypoints_index.cpu().numpy()

            for i in range(len(keypoints_index)):

                im_key = np.ones((self.image_size, self.image_size))
                key_points = self.change2xy(keypoints_index[i])
                self.keypoinsts_list.append(key_points)

                for index in range(len(key_points)):

                    x = [
                        np.clip(key_points[index][0] - radius, 0,
                                self.image_size),
                        np.clip(key_points[index][0] + radius, 0,
                                self.image_size)
                    ]
                    y = [
                        np.clip(key_points[index][1] - radius, 0,
                                self.image_size),
                        np.clip(key_points[index][1] + radius, 0,
                                self.image_size)
                    ]

                    im_key[y[0]:y[1] + 1, key_points[index][0]] = 0
                    im_key[key_points[index][1], x[0]:x[1] + 1] = 0

                self.keypoints_row.append(im_key)

        if self.visualize_gui:

            tactile_color[1] = torch.flip(tactile_color[1], [1])
            self.flip_static_color = self.static_color.clone()

            self.flip_static_color[1] = torch.flip(self.static_color[1], [1])

            color_n_depth = self.updateGUI(tactile_color, tactile_depth)

        tacto_color, tacto_depth = None, None

        #tacto
        if self.render_tacto:
            tactile_state = tactile_state.cpu().numpy()
            tacto_color, tacto_depth = self.visulize_tacto(tactile_state)

        #sapein
        if self.render_sapien_camera:
            self.update_sapien_render()  # update sapein rendering

        #make video
        if self.make_video:
            self.capture_video(tactile_color.cpu().numpy(),
                               tactile_depth.cpu().numpy(), tacto_color,
                               tacto_depth)

        return color_n_depth

    def change2xy(self, keypoints):
        xy = keypoints[:, :2]
        xy[:, 0] = np.round((1 + xy[:, 0]) * (self.image_size - 1) / 2)
        xy[:, 1] = np.round((1 - xy[:, 1]) * (self.image_size - 1) / 2)
        return np.uint8(xy)

    #==============================================================
    # modify signgle depth channel to three channel
    #==============================================================
    def _depth_to_color(self,
                        depth: np.ndarray,
                        zrange: float = 0.002) -> List:
        "tactile depth modification to three channel images"

        gray = (np.clip(depth / zrange, 0, 1) * 255).astype(np.uint8)
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    #==============================================================
    # capture the video for rendering result
    #==============================================================

    def capture_video(self, tactile_color: np.ndarray,
                      tactile_depth: np.ndarray, tacto_color: np.ndarray,
                      tacto_depth: np.ndarray) -> None:
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
    # init sapien view rendering
    #==============================================================
    def init_sapien_camera(self) -> None:
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
            './video/simulation3d_sapien.mp4', fps=60)

    #==============================================================
    # update sapien rendering
    #==============================================================
    def update_sapien_render(self) -> None:
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
    # visulize pytorch3d rendering result
    #==============================================================
    def visulize_tactile(self, decreased_cameras_position:torch.Tensor, decreased_cameras_quat:torch.Tensor, decreased_obj_position,\
        decreased_obj_quat:torch.Tensor, decreased_contact_forces:torch.Tensor,obj_indexes:torch.Tensor,obj_scales:torch.Tensor)->Union[torch.Tensor,torch.Tensor]:

        color, depth = self.renderer.render(decreased_obj_position,
                                            decreased_obj_quat,
                                            decreased_cameras_position,
                                            decreased_cameras_quat,
                                            decreased_contact_forces,
                                            obj_indexes, obj_scales)

        return color, depth

    #==============================================================
    # update pyrender rendering
    #==============================================================
    def update_tacto(self, cameras_position: np.ndarray,
                     cameras_quat: np.ndarray, obj_position: np.ndarray,
                     obj_quat: np.ndarray, contact_forces: np.ndarray,
                     force_index: np.ndarray) -> Union[np.ndarray, np.ndarray]:
        '''
        update tacto rendering
        '''

        # start = time.time()
        color, depth = self.digits_tacto.render(cameras_position, cameras_quat,
                                                obj_position, obj_quat,
                                                contact_forces, force_index)
        # end = time.time()
        # print("tacto time:", end - start)

        return color, depth

    #==============================================================
    # update tacto rendering
    #==============================================================
    def visulize_tacto(
            self, tactile_states: np.ndarray) -> Union[np.ndarray, np.ndarray]:

        contact_forces = tactile_states[:, -2].reshape(-1, 1)

        cameras_position = tactile_states[:, :3].reshape(-1, 3)
        cameras_quat = tactile_states[:, 3:7].reshape(-1, 4)
        obj_position = tactile_states[:, 7:10].reshape(-1, 3)
        obj_quat = tactile_states[:, 10:14].reshape(-1, 4)

        force_index, _ = np.where(contact_forces > 1e-2)

        if np.max(contact_forces) < 1e-2:

            force_index = None

        tacto_color, tacto_depth = self.update_tacto(
            cameras_position[force_index], cameras_quat[force_index],
            obj_position[force_index], obj_quat[force_index],
            contact_forces[force_index], force_index)  # update tacto rendering

        return tacto_color, tacto_depth

    #==============================================================
    # render image from tacto
    #==============================================================

    def render_tactile_from_tacto(
            self, tactile_states: np.ndarray) -> Union[np.ndarray, np.ndarray]:
        """
        Render tacto images from each camera's view.
        """

        tacto_color, tacto_depth = self.visulize_tacto(tactile_states)

        # transpose images in the numpy version
        tacto_color = np.array(tacto_color)
        imgs = tacto_color.reshape(-1, 4, tacto_color.shape[1],
                                   tacto_color.shape[1], 3)
        imgs = np.transpose(imgs, (0, 1, 4, 2, 3))
        imgs = np.array(imgs).reshape(-1, 12, tacto_color.shape[1],
                                      tacto_color.shape[1])

        return imgs[0] / 255

    #==============================================================
    # reset objects index
    #==============================================================
    def update_obj_index(self, obj_indexes: int) -> None:
        if not self.any_train:
            self.obj_index = 0
        else:

            self.obj_index = obj_indexes

        if self.render_tacto:

            self.digits_tacto.update_object([
                self.objects_mesh[self.obj_index][0].cpu().numpy(),
                self.objects_mesh[self.obj_index][1].cpu().numpy()
            ])

    #==============================================================
    # make mask
    #==============================================================
    def make_mask(self, color):

        diff = color - self.rgb[0, :, :, :3]

        mask = torch.where(torch.sum(abs(diff), dim=3) > 1e-3, 0, 1)
        return mask[:, :, :, None]

    #==============================================================
    # transfer depth
    #==============================================================
    def make_depth(self, depth) -> torch.Tensor:
        "change the depth"
        zrange = 0.002
        gray = torch.clip(depth / zrange, 0, 1)

        return gray

    #==============================================================
    # reset for previous image batch
    #==============================================================
    def reset(self, tactile_states, eval: bool = False) -> torch.Tensor:
        self.render(tactile_states, reset=True)
        if eval:
            self.pre_image_batch_eval = self.visual_color.clone()
            return self.pre_image_batch_eval * 0

        self.pre_image_batch = self.visual_color.clone()
        return self.pre_image_batch * 0

    #==============================================================
    # render diff images
    #==============================================================
    def render_diff(self,
                    output: torch.Tensor,
                    reset: bool = False,
                    eval: bool = False):

        if reset:
            self.diff_imgs = output.clone()

            if eval:  # for eval during wandb
                self.pre_image_batch_eval = output.clone()
                return self.transpose_imgs(output) * 0

            self.pre_image_batch = output.clone()
            return self.transpose_imgs(output) * 0

        if eval:  # for eval during wandb
            imgs = output - self.pre_image_batch_eval
            self.pre_image_batch_eval = output.clone()
            return self.transpose_imgs(imgs)

        imgs = output - self.static_color
        diff = imgs + 0.5
        diff[diff < 0.5] = (diff[diff < 0.5] - 0.5) * 0.7 + 0.5
        self.diff_imgs = torch.mean(torch.abs(diff - 0.5), axis=-1)[:, :, :,
                                                                    None]

        self.diff_imgs = ((torch.abs(self.diff_imgs) > 0.15))
        self.diff_imgs = torch.cat(
            [self.diff_imgs, self.diff_imgs, self.diff_imgs], axis=3)

        self.pre_image_batch = output.clone()

        imgs = self.transpose_imgs(imgs)

        return imgs

    #==============================================================
    # render diff images
    #==============================================================

    def render_opticalflow(self,
                           output: torch.Tensor,
                           reset: bool = False,
                           eval: bool = False):

        if reset:
            if eval:  # for eval during wandb
                self.pre_image_batch_eval = output.clone()
                return self.transpose_imgs(output) * 0

            self.pre_image_batch = output.clone()

            return self.transpose_imgs(output) * 0

        if eval:  # for eval during wandb
            imgs = self.render_opticalflow_image(output, self.pre_image_batch)
            imgs = self.transpose_imgs(imgs)
            self.pre_image_batch_eval = output.clone()
            return self.transpose_imgs(imgs)

        imgs = self.render_opticalflow_image(output, self.pre_image_batch)
        dim = imgs.shape[1]

        resize_images = (torchvision.transforms.Resize(
            size=(self.image_size, self.image_size))(imgs))

        imgs = resize_images.reshape(-1, self.num_cameras * dim,
                                     self.image_size, self.image_size)

        # imgs = self.transpose_imgs(imgs)
        self.pre_image_batch = output.clone()

        return imgs

    def random_pad(self, output):

        nonzero_indices = torch.nonzero(output)
        pad_index = torch.randint(0, len(nonzero_indices), (int(
            len(nonzero_indices) *
            np.random.uniform(low=0.05, high=0.2, size=(1, ))[0]), ))
        #pad_index = np.random.randint(0, len(nonzero_indices),int(len(nonzero_indices) * 0.05 + 1))

        index = nonzero_indices[pad_index]

        output[index[:, 0], index[:, 1], index[:, 2], index[:, 3]] = 0

        index[:, [2]] = torch.clip(index[:, [2]] + 1, 0, 63)  #(0,1)
        output[index[:, 0], index[:, 1], index[:, 2], index[:, 3]] = 0

        index[:, [1]] = torch.clip(index[:, [1]] + 1, 0, 63)  #(1,1)
        output[index[:, 0], index[:, 1], index[:, 2], index[:, 3]] = 0

        index[:, [2]] = torch.clip(index[:, [2]] - 1, 0, 63)  #(1,0)
        output[index[:, 0], index[:, 1], index[:, 2], index[:, 3]] = 0

        return output

    #==============================================================
    # render image from pytorch3d
    #==============================================================
    def render(self,
               tactile_states: np.ndarray,
               reset: bool = False,
               eval: bool = False) -> torch.Tensor:
        """
        Render tacto images from each camera's view.
        """

        tactile_states = torch.as_tensor(tactile_states,
                                         device=self.device,
                                         dtype=torch.float32)

        output_color, _ = self.render_static(len(tactile_states))  #(N,64,64,3)
        self.static_color = output_color.clone()

        # tactile_states[:, -2] +=10

        contact_forces = tactile_states[:, 14].reshape(-1, 1)
        self.visual_color = output_color

        self.visual_depth = torch.zeros(
            (len(tactile_states), output_color.shape[1], output_color.shape[2],
             1),
            device=self.device,
            dtype=torch.float32)

        #judge the output result

        if self.use_depth and not self.use_rgb:  #only depth
            output = torch.zeros((len(tactile_states), output_color.shape[1],
                                  output_color.shape[2], 1),
                                 device=self.device,
                                 dtype=torch.float32)

        elif self.use_depth and self.use_rgb and self.use_mask:  #RGB+depth+mask

            output = torch.cat(
                (output_color * (1.0),
                 torch.zeros((len(tactile_states), output_color.shape[1],
                              output_color.shape[2], 2),
                             device=self.device,
                             dtype=torch.float32)),
                dim=3)

        elif self.use_depth and self.use_rgb and not self.use_mask:  #RGB+depth

            output = torch.cat(
                (output_color * (1.0),
                 torch.zeros((len(tactile_states), output_color.shape[1],
                              output_color.shape[2], 1),
                             device=self.device,
                             dtype=torch.float32)),
                dim=3)

        elif not self.use_depth and self.use_rgb and self.use_mask:  #RGB+Mask

            output = torch.cat(
                (output_color * (1.0),
                 torch.zeros((len(tactile_states), output_color.shape[1],
                              output_color.shape[2], 1),
                             device=self.device,
                             dtype=torch.float32)),
                dim=3)

        elif not self.use_depth and self.use_rgb and not self.use_mask:
            output = output_color * (1.0)

        elif not self.use_depth and not self.use_rgb and self.use_mask:  #mask only
            output = torch.zeros((len(tactile_states), output_color.shape[1],
                                  output_color.shape[2], 1),
                                 device=self.device,
                                 dtype=torch.float32)

        if torch.max(contact_forces) < 1e-2:  #no images need to render
            self.visual_color = output

            if self.optical_flow:

                return self.render_opticalflow(output, reset, eval)

            if self.use_diff:

                imgs = self.render_diff(output, reset, eval)
                return imgs

            imgs = self.transpose_imgs(output)

            return imgs






        decreased_cameras_position, decreased_cameras_quat, decreased_obj_position,\
        decreased_obj_quat, decreased_contact_forces, force_index,obj_indexes,obj_scales=self.split_state(tactile_states)

        color, depth = self.renderer.render(decreased_obj_position,
                                            decreased_obj_quat,
                                            decreased_cameras_position,
                                            decreased_cameras_quat,
                                            decreased_contact_forces,
                                            obj_indexes, obj_scales)

        # self.visual_color = torch.zeros((len(tactile_states), output_color.shape[1],
        #                           output_color.shape[2], 3),
        #                          device=self.device,
        #                          dtype=torch.float32)
        self.visual_color[force_index] = color

        self.visual_depth[force_index] = depth

        #judge the output result
        if self.use_depth and not self.use_rgb:  #only depth

            # depth = self.make_depth(depth)
            imgs = color - self.static_color[:len(color)]

            diff = imgs + 0.5
            diff[diff < 0.5] = (diff[diff < 0.5] - 0.5) * 0.7 + 0.5
            depth = torch.mean(torch.abs(diff - 0.5), axis=-1)[:, :, :, None]
            output[force_index] = depth

            if self.crop:  # normalized
                output[force_index] = depth / torch.clip(
                    torch.max(depth.view(depth.shape[0], -1), dim=1).values,
                    0.05, 1)[:, None, None, None] * torch.rand(
                        depth.shape[0], device=self.device)[:, None, None,
                                                            None]
            else:
                output = output

            return self.transpose_imgs(output)

        elif self.use_depth and self.use_rgb and self.use_mask:  #RGB+depth+mask
            mask = self.make_mask(color)
            depth = self.make_depth(depth)
            output[force_index] = torch.cat((color, depth, mask), dim=3)
            return self.transpose_imgs(output)

        elif self.use_depth and self.use_rgb and not self.use_mask:  #RGB+depth
            depth = self.make_depth(depth)
            output[force_index] = torch.cat((color, depth), dim=3)

            return self.transpose_imgs(output)

        elif not self.use_depth and self.use_rgb and self.use_mask:  #RGB+Mask
            mask = self.make_mask(color)
            output[force_index] = torch.cat((color, mask), dim=3)

            return self.transpose_imgs(output)

        elif not self.use_depth and not self.use_rgb and self.use_mask:  #mask only
            imgs = color - self.static_color[:len(color)]
            diff = imgs + 0.5
            diff[diff < 0.5] = (diff[diff < 0.5] - 0.5) * 0.7 + 0.5
            depth = torch.mean(torch.abs(diff - 0.5), axis=-1)[:, :, :, None]

            mask = torch.as_tensor((torch.abs(depth) > 0.06),
                                   dtype=torch.float32)
            output[force_index] = mask
            if self.crop:  # add noise
                output[force_index] = mask * torch.rand(
                    mask.shape[0], device=self.device)[:, None, None, None]

            # mean = 0.0
            # std = 0.05
            # noise = torch.randn_like(output) * std + mean
            # output = output + noise

            return self.transpose_imgs(output)

        else:  # other case just output rgb image
            output[force_index] = color
            self.visual_color = output

            if self.optical_flow:

                return self.render_opticalflow(output, reset, eval)

            if self.use_diff:
                imgs = self.render_diff(output, reset, eval)

                return imgs

            imgs = self.transpose_imgs(output)

            return imgs
