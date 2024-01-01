from functools import cached_property
from typing import Optional, ClassVar

import numpy as np
import sapien.core as sapien
import transforms3d
from gym.utils import seeding
from sapien.utils import Viewer

from hand_teleop.env.rl_env.base import BaseRLEnv

from hand_teleop.env.sim_env.relocate_env import RelocateEnv, LabRelocateEnv
from hand_teleop.real_world import lab
#=====================tactile===========================
from hand_teleop.utils.tactile_utils import obtain_tactile_postion, obtain_tactile_force, Tactile
import pdb
from hand_teleop.utils.render_scene_utils import set_entity_color
# import torch
# from hand_teleop.utils.sapien_render import fetch_tactile_image
#=====================tactile===========================
OBJECT_LIFT_LOWER_LIMIT = -0.03


class RelocateRLEnv(RelocateEnv, BaseRLEnv):

    def __init__(self,
                 use_gui=False,
                 frame_skip=5,
                 robot_name="adroit_hand_free",
                 constant_object_state=False,
                 rotation_reward_weight=0,
                 object_category="YCB",
                 object_name="tomato_soup_can",
                 object_scale=1.0,
                 randomness_scale=1,
                 friction=1,
                 object_pose_noise=0.01,
                 **renderer_kwargs):
        super().__init__(use_gui, frame_skip, object_category, object_name,
                         object_scale, randomness_scale, friction,
                         **renderer_kwargs)
        self.setup(robot_name)
        self.constant_object_state = constant_object_state
        self.rotation_reward_weight = rotation_reward_weight
        self.object_pose_noise = object_pose_noise

        # Parse link name
        self.palm_link_name = self.robot_info.palm_name
        self.palm_link = [
            link for link in self.robot.get_links()
            if link.get_name() == self.palm_link_name
        ][0]

        # Object init pose
        self.object_episode_init_pose = sapien.Pose()

    def get_oracle_state(self):
        robot_qpos_vec = self.robot.get_qpos()
        object_pose = self.object_episode_init_pose if self.constant_object_state else self.manipulated_object.get_pose(
        )
        object_pose_vec = np.concatenate([object_pose.p, object_pose.q])
        palm_pose = self.palm_link.get_pose()
        target_in_object = self.target_pose.p - object_pose.p
        target_in_palm = self.target_pose.p - palm_pose.p
        object_in_palm = object_pose.p - palm_pose.p
        palm_v = self.palm_link.get_velocity()
        palm_w = self.palm_link.get_angular_velocity()
        theta = np.arccos(
            np.clip(
                np.power(np.sum(object_pose.q * self.target_pose.q), 2) * 2 -
                1, -1 + 1e-8, 1 - 1e-8))
        return np.concatenate([
            robot_qpos_vec, object_pose_vec, palm_v, palm_w, object_in_palm,
            target_in_palm, target_in_object, self.target_pose.q,
            np.array([theta])
        ])

    def get_robot_state(self):
        robot_qpos_vec = self.robot.get_qpos()
        palm_pose = self.palm_link.get_pose()
        return np.concatenate([
            robot_qpos_vec, palm_pose.p, self.target_pose.p, self.target_pose.q
        ])

    def get_reward(self, action):

        object_pose = self.manipulated_object.get_pose()
        palm_pose = self.palm_link.get_pose()
        is_contact = self.check_contact(self.robot_collision_links,
                                        [self.manipulated_object])

        reward = -0.1 * min(np.linalg.norm(palm_pose.p - object_pose.p), 0.5)
        if is_contact:
            reward += 0.1
            lift = min(object_pose.p[2],
                       self.target_pose.p[2]) - self.object_height
            lift = max(lift, 0)
            reward += 5 * lift
            if lift > 0.015:
                reward += 2
                obj_target_distance = min(
                    np.linalg.norm(object_pose.p - self.target_pose.p), 0.5)
                reward += -1 * min(
                    np.linalg.norm(palm_pose.p - self.target_pose.p), 0.5)
                reward += -3 * obj_target_distance  # make object go to target

                if obj_target_distance < 0.1:
                    reward += (0.1 - obj_target_distance) * 20
                    theta = np.arccos(
                        np.clip(
                            np.power(
                                np.sum(object_pose.q * self.target_pose.q), 2)
                            * 2 - 1, -1 + 1e-8, 1 - 1e-8))
                    reward += max(
                        (np.pi / 2 - theta) * self.rotation_reward_weight, 0)
                    if theta < np.pi / 4 and self.rotation_reward_weight >= 1e-6:
                        reward += (np.pi / 4 -
                                   theta) * 6 * self.rotation_reward_weight

        return reward

    def reset(self,
              *,
              seed: Optional[int] = None,
              return_info: bool = False,
              options: Optional[dict] = None):
        super().reset(seed=seed)

        if not self.is_robot_free:
            qpos = np.zeros(self.robot.dof)
            xarm_qpos = self.robot_info.arm_init_qpos
            qpos[:self.arm_dof] = xarm_qpos
            self.robot.set_qpos(qpos)

            self.robot.set_drive_target(qpos)
            init_pos = np.array(lab.ROBOT2BASE.p) + self.robot_info.root_offset
            init_pose = sapien.Pose(init_pos,
                                    transforms3d.euler.euler2quat(0, 0, 0))
        else:
            init_pose = sapien.Pose(
                np.array([-0.4, 0, 0.2]),
                transforms3d.euler.euler2quat(0, np.pi / 2, 0))
        self.robot.set_pose(init_pose)
        self.reset_internal()
        self.object_episode_init_pose = self.manipulated_object.get_pose()
        random_quat = transforms3d.euler.euler2quat(
            *(self.np_random.randn(3) * self.object_pose_noise * 10))
        random_pos = self.np_random.randn(3) * self.object_pose_noise
        self.object_episode_init_pose = self.object_episode_init_pose * sapien.Pose(
            random_pos, random_quat)
        obs = self.get_observation()

        return obs

    @cached_property
    def obs_dim(self):
        if not self.use_visual_obs:
            return self.robot.dof + 7 + 6 + 9 + 4 + 1
        else:
            return len(self.get_robot_state())

    def is_done(self):
        return False

    @cached_property
    def horizon(self):
        return 250


class LabArmAllegroRelocateRLEnv(LabRelocateEnv, BaseRLEnv):

    def __init__(self,
                 use_gui=False,
                 frame_skip=10,
                 robot_name="allegro_hand_xarm6_wrist_mounted_face_front",
                 rotation_reward_weight=0,
                 object_category="YCB",
                 object_name="tomato_soup_can",
                 randomness_scale=1,
                 friction=1,
                 root_frame="robot",
                 use_tips=False,
                 use_buffer=False,
                 reduced_state=False,
                 **renderer_kwargs):
        # if "allegro" not in robot_name or "free" in robot_name:
        #     raise ValueError(
        #         f"Robot name: {robot_name} is not valid xarm allegro robot.")

        super().__init__(use_gui, frame_skip, object_category, object_name,
                         randomness_scale, friction, **renderer_kwargs)

        # Base class
        self.setup(robot_name)
        self.rotation_reward_weight = rotation_reward_weight

        # Parse link name
        self.palm_link_name = self.robot_info.palm_name
        self.palm_link = [
            link for link in self.robot.get_links()
            if link.get_name() == self.palm_link_name
        ][0]

        # Base frame for observation
        self.root_frame = root_frame
        self.base_frame_pos = np.zeros(3)
        self.use_tips = use_tips
        self.use_buffer = use_buffer
        self.reduced_state = reduced_state

        #Finger tip: thumb, index, middle, ring
        # finger_tip_names = [
        #     "link_15.0_tip_fsr", "link_15.0_fsr", "link_14.0_fsr",
        if "sfr" in self.robot_name:

            finger_tip_names = [
                "link_15.0_tip", "link_3.0_tip", "link_7.0_tip",
                "link_11.0_tip"
            ]

            finger_contact_link_name = [
                "link_15.0_tip_fsr", "link_15.0_fsr", "link_14.0_fsr",
                "link_3.0_tip_fsr", "link_2.0_fsr", "link_1.0_fsr",
                "link_7.0_tip_fsr", "link_6.0_fsr", "link_5.0_fsr",
                "link_11.0_tip_fsr", "link_10.0_fsr", "link_9.0_fsr",
                "link_4.0_fsr", "link_8.0_fsr", "link_0.0_fsr", "link_13.0_fsr"
            ]

        else:
            finger_tip_names = [
                "link_15.0_tip", "link_3.0_tip", "link_7.0_tip",
                "link_11.0_tip"
            ]

            finger_contact_link_name = [
                "link_15.0_tip", "link_15.0", "link_14.0", "link_3.0_tip",
                "link_3.0", "link_2.0", "link_1.0", "link_7.0_tip", "link_7.0",
                "link_6.0", "link_5.0", "link_11.0_tip", "link_11.0",
                "link_10.0", "link_9.0"
            ]

        #==========================contact =================================
        finger_box_names = [
            "link_15.0_tip", "link_3.0_tip", "link_7.0_tip", "link_11.0_tip"
        ]
        #==========================contact =================================

        self.render_link = None
        if self.object_category == "partnet":

            manipulated_object_link_names = [
                link.get_name()
                for link in self.manipulated_object.get_links()
            ]
            render_link_name = np.loadtxt(
                "./assets/partnet-mobility-dataset/render_mesh/" +
                self.object_name + "/link.txt",
                dtype=str)
            self.render_link = [
                self.manipulated_object.get_links()[
                    manipulated_object_link_names.index(name)]
                for name in render_link_name
            ]
        #=====================tactile===========================

        robot_link_names = [link.get_name() for link in self.robot.get_links()]
        self.finger_tip_links = [
            self.robot.get_links()[robot_link_names.index(name)]
            for name in finger_tip_names
        ]
        self.finger_contact_links = [
            self.robot.get_links()[robot_link_names.index(name)]
            for name in finger_contact_link_name
        ]

        #==========================contact =================================
        #check the finger box for penalty
        self.finger_box_links = [
            self.robot.get_links()[robot_link_names.index(name)]
            for name in finger_box_names
        ]
        #==========================contact =================================

        #=====================tactile===========================
        '''
        TODO:Check whether need to modify
        '''

        if "sfr" in self.robot_name:
            self.finger_contact_ids = np.array([0] * 3 + [1] * 3 + [2] * 3 +
                                               [3] * 3 + [4] * 4)
        else:
            self.finger_contact_ids = np.array([0] * 3 + [1] * 4 + [2] * 4 +
                                               [3] * 4 + [4])

        #==============================contact============================

        self.finger_tip_pos = np.zeros([len(finger_tip_names), 3])
        self.finger_reward_scale = np.ones(len(self.finger_tip_links)) * 0.01
        self.finger_reward_scale[0] = 0.04

        # Object, palm, target pose
        self.object_pose = self.manipulated_object.get_pose()
        self.palm_pose = self.palm_link.get_pose()
        self.palm_pos_in_base = np.zeros(3)
        self.object_in_tip = np.zeros([len(finger_tip_names), 3])
        self.target_in_object = np.zeros([3])
        self.target_in_object_angle = np.zeros([1])
        self.object_lift = 0
        #==============================contact============================

        # Contact buffer
        self.robot_object_contact = np.zeros(len(finger_tip_names) +
                                             1)  # four tip,
        self.contact_boolean = np.zeros(len(self.finger_contact_links) + 1)

        #==============================contact============================

        self.num = 0

    def update_cached_state(self):
        for i, link in enumerate(self.finger_tip_links):
            self.finger_tip_pos[i] = self.finger_tip_links[i].get_pose().p

        if "sfr" in self.robot_name:
            check_contact_links = self.finger_contact_links
        else:
            check_contact_links = self.finger_contact_links + [self.palm_link]

        self.contact_boolean, force = self.check_actor_pair_contacts(
            check_contact_links, self.manipulated_object)

        self.robot_object_contact[:] = np.clip(
            np.bincount(self.finger_contact_ids, weights=self.contact_boolean),
            0, 1)
        self.object_pose = self.manipulated_object.get_pose()
        self.palm_pose = self.palm_link.get_pose()
        self.palm_pos_in_base = self.palm_pose.p - self.base_frame_pos
        self.object_in_tip = self.object_pose.p[None, :] - self.finger_tip_pos

        self.object_lift = self.object_pose.p[2] - self.object_height
        self.target_in_object = self.target_pose.p - self.object_pose.p
        self.target_in_object_angle[0] = np.arccos(
            np.clip(
                np.power(np.sum(self.object_pose.q * self.target_pose.q), 2) *
                2 - 1, -1 + 1e-8, 1 - 1e-8))

    def get_oracle_state(self):
        robot_qpos_vec = self.robot.get_qpos()

        if self.use_buffer:

            return np.concatenate([
                robot_qpos_vec, self.palm_pos_in_base,
                self.robot_object_contact,
                self.init_obj_pose - self.base_frame_pos,
                self.target_pose.p - self.base_frame_pos, self.target_pose.q
            ])

        elif self.reduced_state:
            return np.concatenate([
                robot_qpos_vec, self.palm_pos_in_base,
                self.init_obj_pose - self.base_frame_pos,
                self.target_pose.p - self.base_frame_pos, self.target_pose.q
            ])

        else:
            # return np.concatenate([
            #     robot_qpos_vec, self.palm_pos_in_base,
            #     self.init_obj_pose - self.base_frame_pos,
            #     self.target_pose.p - self.base_frame_pos, self.target_pose.q
            # ])

            object_pos = self.object_pose.p
            object_quat = self.object_pose.q
            object_pose_vec = np.concatenate(
                [object_pos - self.base_frame_pos, object_quat])
            robot_qpos_vec = self.robot.get_qpos()
            return np.concatenate([
                robot_qpos_vec,
                self.palm_pos_in_base,  # dof + 3
                object_pose_vec,
                self.object_in_tip.flatten(),
                self.robot_object_contact,  # 7 + 12 + 5
                self.target_in_object,
                self.target_pose.q,
                self.target_in_object_angle  # 3 + 4 + 1
            ])

    def get_tactile_state(self):
        '''
        get the realted information for tactile
        '''

        # position, quaternion, force
        tactile_contact_force = obtain_tactile_force(
            self.scene, [self.manipulated_object],
            self.finger_tip_links).reshape(-1, 1)
        tactile_pose = obtain_tactile_postion(self.manipulated_object,
                                              self.finger_tip_links,
                                              render_link=self.render_link)

        return np.concatenate(
            (
                tactile_pose,
                tactile_contact_force,
                np.repeat(self.obj_index, 4).reshape(4, -1),
            ),
            axis=1)  # robot postion,object position, contact force

    #=====================tactile===========================

    def get_robot_state(self):
        robot_qpos_vec = self.robot.get_qpos()

        return np.concatenate([
            robot_qpos_vec, self.palm_pos_in_base,
            self.init_obj_pose - self.base_frame_pos,
            self.target_pose.p - self.base_frame_pos, self.target_pose.q
        ])

    def get_reward(self, action):

        finger_object_dist = np.linalg.norm(self.object_in_tip,
                                            axis=1,
                                            keepdims=False)
        finger_object_dist = np.clip(finger_object_dist, 0.03, 0.8)
        reward = np.sum(1.0 / (0.06 + finger_object_dist) *
                        self.finger_reward_scale)
        # at least one tip and palm or two tips are contacting obj. Thumb contact is required.
        is_contact = np.sum(self.robot_object_contact) >= 2

        if is_contact:

            reward += 0.5
            lift = np.clip(self.object_lift, 0, 0.2)
            reward += 10 * lift
            if lift > 0.02:
                reward += 1
                target_obj_dist = np.linalg.norm(self.target_in_object)
                reward += 1.0 / (0.04 + target_obj_dist)

                if target_obj_dist < 0.1:
                    theta = self.target_in_object_angle[0]
                    reward += 4.0 / (0.4 + theta) * self.rotation_reward_weight

        action_penalty = np.sum(np.clip(self.robot.get_qvel(), -1, 1)**
                                2) * -0.01
        controller_penalty = (self.cartesian_error**2) * -1e3

        return (reward + action_penalty + controller_penalty) / 10 + np.sum(
            self.robot_object_contact *
            [2, 1, 1, 1, 2]) / 7  #- palm_self.contact_boolean - np.sum(
        #    box_object_contact) / 2  #more contact on gel, less contact on box

    @cached_property
    def obs_dim(self):
        if not self.use_visual_obs:
            return len(self.get_oracle_state())
        else:
            return len(self.get_robot_state())

    def reset(self,
              *,
              seed: Optional[int] = None,
              return_info: bool = False,
              options: Optional[dict] = None):
        # Gym reset function
        if seed is not None:
            self.seed(seed)

        self.reset_internal()
        # Set robot qpos
        qpos = np.zeros(self.robot.dof)
        xarm_qpos = self.robot_info.arm_init_qpos

        qpos[:self.arm_dof] = xarm_qpos

        self.robot.set_qpos(qpos)
        self.robot.set_drive_target(qpos)

        # qpos = self.reset_robot()
        # self.robot.set_qpos(qpos)
        # self.robot.set_drive_target(qpos)
        # self.robot.set_drive_velocity_target(qpos * 0)
        # self.robot.set_root_velocity(qpos * 0)

        self.num = 0

        # Set robot pose
        init_pos = np.array(lab.ROBOT2BASE.p) + self.robot_info.root_offset
        init_pose = sapien.Pose(init_pos,
                                transforms3d.euler.euler2quat(0, 0, 0))
        self.robot.set_pose(init_pose)

        if self.object_category == "egad" or self.object_category == "assembling_kits":
            for _ in range(2):

                self.step(np.zeros(self.robot.dof))

            self.object_height = self.manipulated_object.get_pose().p[2]
            self.init_obj_pose = self.manipulated_object.get_pose().p

        if self.root_frame == "robot":
            self.base_frame_pos = self.robot.get_pose().p

        elif self.root_frame == "world":
            self.base_frame_pos = np.zeros(3)
        else:
            raise NotImplementedError

        self.update_cached_state()
        self.update_imagination(reset_goal=True)

        obs = self.get_observation()

        return obs

    def is_done(self):
        return self.object_lift < OBJECT_LIFT_LOWER_LIMIT

    @cached_property
    def horizon(self):
        return 200

    def reset_robot(self):

        palm_pose = self.init_obj_pose.copy()
        if "front" in self.robot_name:
            palm_pose[0] += 0.45
            palm_pose[1] += 0.00
            palm_pose[2] -= 0.05
            model = self.robot.create_pinocchio_model()
            from scipy.spatial.transform import Rotation as R
            if "30" in self.robot_name:
                r = R.from_rotvec([np.pi / 3, 0, 0])
            else:
                r = R.from_rotvec([np.pi / 2, 0, 0])
            quat = r.as_quat()[[3, 0, 1, 2]]  #x,y,z,w

        elif "down" in self.robot_name:

            pose = self.init_obj_pose

            palm_pose[0] += 0.55
            palm_pose[1] += pose[1] * 0
            palm_pose[2] += 0.10
            model = self.robot.create_pinocchio_model()
            from scipy.spatial.transform import Rotation as R
            r = R.from_euler('xyz', [0, 90, 0], degrees=True)
            quat = r.as_quat()[[3, 0, 1, 2]]  #x,y,z,w
        target_pose = sapien.Pose(palm_pose, quat)
        result, success, error = model.compute_inverse_kinematics(
            self.palm_link.get_index(), target_pose, self.robot.get_qpos(),
            [1] * 6 + [0] * 16)

        return result

    #=====================tactile===========================

    def fetch_tactile_camera(self):
        self.scene.update_render()
        normals = []
        coords = []

        for cam in self.link_cameras:
            await_dl_list = cam.take_picture_and_get_dl_tensors_async(
                ['Position'])

            depth = cam.get_position_rgba()

            depth[depth[..., 2] < -0.6] = 0
            depth[depth[..., 2] >= 0] = 0

            # pixel_coords = torch.as_tensor(depth[:, :, :3]).reshape(64, 64, 3)
            normal = cam.get_normal_rgba()[:, :, :3]
            # normal[normal[..., 1] > 0.9] = 0

            # pixel_normals = torch.as_tensor(normal).reshape(64, 64, 3)
            normals.append(normal)
            coords.append(depth)

        return torch.as_tensor(normals), torch.as_tensor(coords)

    # update tactile
    def update_tactile(self, tactile):
        tactile_states = self.get_tactile_state()
        tactile.visulize(tactile_states)
        # normals, coords = self.fetch_tactile_camera()
        # fetch_tactile_image(normals, coords)

    #=====================tactile===========================


def main_env():
    from time import time
    env = LabArmAllegroRelocateRLEnv(
        use_gui=True,
        robot_name="xarm7_gripper_front",
        object_name="any_train",
        object_category="02876657",
        frame_skip=10,
        use_visual_obs=False,
        use_orientation=False,
        use_buffer=True)  #(D1,D2,B0,B1,B2,A3,A2,A1,A0)
    base_env = env
    robot_dof = env.robot.dof
    env.seed(0)
    env.reset()

    tic = time()
    env.reset()
    tac = time()
    print(f"Reset time: {(tac - tic) * 1000} ms")

    tic = time()
    # for i in range(1000):
    #     # action = np.random.rand(robot_dof) * 2 - 1
    #     # action[2] = 0.1
    #     obs, reward, done, _ = env.step(action)
    tac = time()
    print(f"Step time: {(tac - tic)} ms")

    viewer = Viewer(base_env.renderer)
    viewer.set_scene(base_env.scene)
    base_env.viewer = viewer
    viewer.set_camera_rpy(r=0, p=0, y=-np.pi)  # change the viewer direction
    # viewer.set_camera_xyz(-1, 0, 1)

    env.reset()

    # viewer.toggle_pause(True)
    pose = env.palm_link.get_pose()

    frame = 0
    while not viewer.closed:
        frame += 1
        action = np.zeros(robot_dof)

        # action[7:] = 0.1
        obs, reward, done, _ = env.step(action)
        env.scene.step()
        if frame % 200 == 0:
            env.reset()
            frame = 0

        #env.simple_step()
        env.render()


if __name__ == '__main__':
    main_env()
