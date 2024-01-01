from functools import cached_property
from typing import Optional, ClassVar

import numpy as np
import sapien.core as sapien
import transforms3d
from gym.utils import seeding
from sapien.utils import Viewer

# from hand_teleop.env.rl_env.base import BaseRLEnv
from hand_teleop.env.rl_env.robot_gripper import RobotEnv
from hand_teleop.env.sim_env.write_env import LabWriteEnv
from hand_teleop.real_world import lab
#=====================tactile===========================
from hand_teleop.utils.tactile_utils import obtain_tactile_postion, obtain_tactile_force, Tactile
import pdb
from transforms3d import quaternions, euler
from scipy.spatial.transform import Rotation as R

# import torch
# from hand_teleop.utils.sapien_render import fetch_tactile_image
#=====================tactile===========================
OBJECT_LIFT_LOWER_LIMIT = -0.03


def quat_delta(q1, q2):
    q_rel = quaternions.qmult(q2, quaternions.qinverse(q1))
    R_rel = quaternions.quat2mat(q_rel)
    theta_x, theta_y, theta_z = euler.mat2euler(R_rel, axes='sxyz')
    return theta_x, theta_y, theta_z


def quat_conjugate(q):
    inv_q = -q
    inv_q[..., 0] *= -1
    return inv_q


def quat_mul(q0, q1):
    assert q0.shape == q1.shape
    assert q0.shape[-1] == 4
    assert q1.shape[-1] == 4

    w0 = q0[..., 0]
    x0 = q0[..., 1]
    y0 = q0[..., 2]
    z0 = q0[..., 3]

    w1 = q1[..., 0]
    x1 = q1[..., 1]
    y1 = q1[..., 2]
    z1 = q1[..., 3]

    w = w0 * w1 - x0 * x1 - y0 * y1 - z0 * z1
    x = w0 * x1 + x0 * w1 + y0 * z1 - z0 * y1
    y = w0 * y1 + y0 * w1 + z0 * x1 - x0 * z1
    z = w0 * z1 + z0 * w1 + x0 * y1 - y0 * x1
    q = np.array([w, x, y, z])
    if q.ndim == 2:
        q = q.swapaxes(0, 1)
    assert q.shape == q0.shape
    return q


class WriteRLEnv(LabWriteEnv, RobotEnv):

    def __init__(self,
                 use_gui=False,
                 frame_skip=10,
                 robot_name="allegro_hand_xarm6_wrist_mounted_face_front",
                 rotation_reward_weight=0,
                 object_category="YCB",
                 object_name="tomato_soup_can",
                 randomness_scale=1,
                 friction=1,
                 use_tactile=False,
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
        self.task_index = 0
        if object_category in ["pen"]:
            self.task_index = 11
        if object_category in ["spoon"]:
            self.task_index = 10
        # elif object_category in ["kitchen"]:
        #     self.task_index = 9

        self.use_tactile = use_tactile

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

        finger_tip_names = ["left_finger_tip", "right_finger_tip"]

        finger_contact_link_name = ["left_finger_tip", "right_finger_tip"]

        finger_box_link_name = ["left_finger_box", "right_finger_box"]

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
        self.finger_box_links = [
            self.robot.get_links()[robot_link_names.index(name)]
            for name in finger_box_link_name
        ]

        #==========================contact =================================
        #check the finger box for penalty
        # self.finger_box_links = [
        #     self.robot.get_links()[robot_link_names.index(name)]
        #     for name in finger_box_names
        # ]
        #==========================contact =================================

        #=====================tactile===========================
        '''
        TODO:Check whether need to modify
        '''

        self.finger_contact_ids = np.array([0] * 1 + [1] * 1 + [2])

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
        self.pre_theta = 0
        self.sign_distance = -1

        # save_articulated_mesh(self.robot, "assets/robot/xarm7_gripper")

    def update_cached_state(self):
        for i, link in enumerate(self.finger_tip_links):
            self.finger_tip_pos[i] = self.finger_tip_links[i].get_pose().p

        check_contact_links = self.finger_contact_links + [self.palm_link]

        if self.object_category in ["pen"]:

            self.contact_boolean, force = self.check_actor_pair_contacts(
                check_contact_links,
                self.manipulated_object.get_links()[-1],
            )
        else:
            self.contact_boolean, force = self.check_actor_pair_contacts(
                check_contact_links,
                self.manipulated_object,
            )
        self.table_object_contact, _ = self.check_actor_pair_contacts(
            [self.manipulated_object], self.tables[0])

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
        theta = self.angles_compare(self.target_pose.q, self.object_pose.q)

        if self.use_buffer:

            return np.concatenate([
                robot_qpos_vec,
                self.palm_pos_in_base,
                self.robot_object_contact[:2],
                (self.init_obj_pose -
                 self.base_frame_pos),  #remove height information
                self.target_pose.p - self.base_frame_pos,
                self.target_pose.q,
                [self.target_angles / 180 * np.pi, theta / 180 * np.pi]
            ])

        else:

            return np.concatenate([
                robot_qpos_vec,
                self.palm_pos_in_base,
                # self.robot_object_contact,
                (self.init_obj_pose - self.base_frame_pos
                 ),  #remove height information
                self.target_pose.p - self.base_frame_pos,
                self.target_pose.q,
                [self.pre_theta / 180 * np.pi]
            ])
        # return np.concatenate([
        #     robot_qpos_vec, self.palm_pos_in_base,
        #     (self.init_obj_pose - self.base_frame_pos),
        #     self.target_pose.p - self.base_frame_pos, self.target_pose.q
        # ])

        # else:
        #     # return np.concatenate([
        #     #     robot_qpos_vec, self.palm_pos_in_base,
        #     #     self.init_obj_pose - self.base_frame_pos,
        #     #     self.target_pose.p - self.base_frame_pos, self.target_pose.q
        #     # ])

        #     object_pos = self.object_pose.p
        #     object_quat = self.object_pose.q
        #     object_pose_vec = np.concatenate(
        #         [object_pos - self.base_frame_pos, object_quat])
        #     robot_qpos_vec = self.robot.get_qpos()
        #     return np.concatenate([
        #         robot_qpos_vec,
        #         self.palm_pos_in_base,  # dof + 3
        #         object_pose_vec,
        #         self.object_in_tip.flatten(),
        #         self.robot_object_contact,  # 7 + 12 + 5
        #         self.target_in_object,
        #         self.target_pose.q,
        #         self.target_in_object_angle  # 3 + 4 + 1
        #     ])

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
        # add more
        return np.concatenate(
            (
                tactile_pose,
                tactile_contact_force,
                np.repeat(self.obj_index, len(self.finger_tip_links)).reshape(
                    len(self.finger_tip_links), -1),
                np.tile(self.scale, len(self.finger_tip_links)).reshape(
                    len(self.finger_tip_links),
                    -1),  # the scale of the objects 
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
        is_contact = np.sum(self.robot_object_contact[:2]) >= 2
        self.target_obj_height_dist = self.manipulated_object.get_pose(
        ).p[2] - self.target_pose.p[2]

        # theta_x, theta_y, theta_z = quat_delta(self.object_pose.q,
        #                                        self.target_pose.q)
        theta = self.angles_compare(self.target_pose.q, self.object_pose.q)

        dist_angle = np.arctan2(self.manipulated_object.get_pose().p[1],
                                self.manipulated_object.get_pose().p[0])

        dist_to_target = dist_angle / 2 * np.pi
        reward += (self.robot_object_contact[0] - 1) * 20

        if is_contact:

            reward += 2

            dist_to_target = np.linalg.norm(self.target_in_object)
            dist_to_target = dist_to_target / self.pre_dist
            # dist_to_target = 1 if dist_to_target > 1 else dist_to_target
            dist_reward = 1 - dist_to_target  #**0.4  # Positive reward [0, 1]

            if dist_reward < 0:

                reward -= dist_reward * 40 * dist_reward
            else:
                reward += dist_reward * 40 * dist_reward

            if abs(theta) < 0.01:

                reward -= abs(theta * 40)
            else:
                reward += (0.1 - abs(theta)) * 10

        action_penalty = np.sum(np.clip(self.robot.get_qvel(), -1, 1)**
                                2) * -0.01
        controller_penalty = (self.cartesian_error**2) * -1e3

        self.overall_rewards += (
            reward + action_penalty + controller_penalty) / 10 + np.sum(
                self.robot_object_contact * [1, 1, -3]) / 2

        return (reward + action_penalty + controller_penalty) / 10 + np.sum(
            self.robot_object_contact *
            [1, 1, -3]) / 2  #- palm_self.contact_boolean - np.sum(
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

        self.overall_rewards = 0

        self.reset_internal()

        # Set robot qpos
        qpos = np.zeros(self.robot.dof)
        xarm_qpos = self.robot_info.arm_init_qpos
        # self.arm_dof = 7
        arm_dof = 7

        qpos[:arm_dof] = xarm_qpos

        self.robot.set_qpos(qpos)
        self.robot.set_drive_target(qpos)

        qpos = self.reset_robot()
        self.robot.set_qpos(qpos)
        self.robot.set_drive_target(qpos)
        self.robot.set_drive_velocity_target(qpos * 0)
        self.robot.set_root_velocity(qpos * 0)

        self.num = 0
        self.reach_height = False

        # Set robot pose
        init_pos = np.array(lab.ROBOT2BASE.p) + self.robot_info.root_offset
        init_pose = sapien.Pose(init_pos,
                                transforms3d.euler.euler2quat(0, 0, 0))
        self.robot.set_pose(init_pose)

        self.initial_dist = np.linalg.norm(
            (self.manipulated_object.get_pose().p -
             self.target_pose.p)[:2]) + abs(
                 (self.manipulated_object.get_pose().p -
                  self.target_pose.p)[2])

        self.initial_xy_dist = np.linalg.norm(
            (self.manipulated_object.get_pose().p - self.target_pose.p)[:2])
        self.pre_dist = np.linalg.norm(self.target_in_object_angle)

        self.pre_theta = self.angles_compare(self.target_pose.q,
                                             self.object_pose.q)
        # self.object_height = self.manipulated_object.get_pose().p[2]
        # self.init_obj_pose = self.manipulated_object.get_pose().p

        if self.root_frame == "robot":
            self.base_frame_pos = self.robot.get_pose().p

        elif self.root_frame == "world":
            self.base_frame_pos = np.zeros(3)
        else:
            raise NotImplementedError

        self.update_cached_state()
        self.update_imagination(reset_goal=True)

        obs = self.get_observation()

        if self.object_category in [
                "pen", "assembling_kits", "spoon", "kitchen"
        ]:
            for i in range(5):
                actions = np.zeros(13)
                actions[7] = 2

                qpos = self.reset_robot()
                self.robot.set_qpos(qpos)
                self.robot.set_drive_target(qpos)
                self.robot.set_drive_velocity_target(qpos * 0)
                self.robot.set_root_velocity(qpos * 0)
                self.manipulated_object.set_pose(self.init_obj_pos)
                self.current_step = 0

        self.pre_theta = self.angles_compare(self.target_pose.q,
                                             self.object_pose.q)
        self.pre_dist = np.linalg.norm(self.target_in_object)

        return obs

    # def angles_compare(self, target_quat, current_quat):

    #     r = R.from_quat(target_quat[[1, 2, 3, 0]])
    #     target_angles = r.as_euler('xyz', degrees=True)[1]
    #     if target_angles < 0:
    #         target_angles = 90 + abs(target_angles)

    #     r = R.from_quat(current_quat[[1, 2, 3, 0]])
    #     current_angles = r.as_euler('xyz', degrees=True)[1]
    #     if current_angles < 0:
    #         current_angles = 90 + abs(current_angles)

    #     delta_angles = target_angles - current_angles

    #     return delta_angles

    def angles_compare(self, target_quat, current_quat):

        r = R.from_quat(target_quat[[1, 2, 3, 0]])
        target_angles = r.as_euler('xyz', degrees=True)[1]
        if target_angles < 0:
            self.target_angles = 90 + abs(target_angles)
        else:
            self.target_angles = 90 - target_angles

        r = R.from_quat(current_quat[[1, 2, 3, 0]])
        current_angles = r.as_euler('xyz', degrees=True)[1]
        if current_angles < 0:
            current_angles = 90 + abs(current_angles)
        else:
            current_angles = 90 - current_angles

        ee_quat = self.palm_link.get_pose().q
        r = R.from_quat(ee_quat[[1, 2, 3, 0]])
        ee_angles = r.as_euler('xyz', degrees=True)[2]

        if ee_angles < 0:
            ee_angles = 360 + ee_angles

        if self.index == 2:
            #target angles
            r = R.from_quat(target_quat[[1, 2, 3, 0]])
            target_angles = r.as_euler('xyz', degrees=True)[1]
            self.target_angles = 90 + abs(90 + target_angles)

            # current angles
            current_angles = 360 - (ee_angles - (90 - current_angles))

            # target_obj_angles = target_angles - ee_angles + 90

            # r = R.from_euler('xyz', [0, -target_obj_angles, 0], degrees=True)
            # orientation = r.as_quat()[[3, 0, 1, 2]]  #x,y,z,w
            # position = np.array([
            #     self.sign_distance * self.object_height * 0.5, 0,
            #     self.object_height * abs(np.cos((target_angles) / 180 * np.pi))
            # ])
            # pose = sapien.Pose(position, orientation)
            # self.target_object.set_pose(pose)
        delta_angles = self.target_angles - current_angles

        return delta_angles

    def is_done(self):

        return (
            (self.manipulated_object.get_pose().p[2] < self.object_height)
            and np.sum(self.robot_object_contact[:2]) < 1
        )  # or (self.manipulated_object.get_pose().p[2]<0)  # lose contact
        # return False

    @cached_property
    def horizon(self):
        return 200

    def reset_robot(self):

        palm_pose = self.manipulated_object.get_pose().p

        palm_pose[0] += 0.55 + self.object_height * np.sin(
            self.rotation_factor * np.pi / 2)
        palm_pose[1] += 0
        palm_pose[2] = self.object_height + 2 * 0 + 0.04

        if self.object_category in ["pen"]:
            palm_pose = self.manipulated_object.get_links()[0].get_pose().p
            palm_pose[0] += 0.55
            palm_pose[2] += (self.object_height / 2 + 0.05) * np.cos(
                self.rotation_factor * np.pi / 2)

        elif self.object_category in ["spoon"]:

            palm_pose = self.manipulated_object.get_pose().p
            palm_pose[0] += 0.55 + (self.object_height * 1.0 + 0.05) * np.sin(
                self.rotation_factor * np.pi / 2) + 0.02
            # + (self.object_height * 1.0 + 0.05) * np.sin(
            #     self.sign_distance * self.rotation_factor * np.pi /
            #     2) - 0.02 * self.sign_distance

            palm_pose[2] = self.manipulated_object.get_pose().p[2] + 0.05

        elif self.object_category in ["kitchen"]:
            palm_pose = self.manipulated_object.get_pose().p
            palm_pose[0] += 0.55 + (self.object_height * 1.0 + 0.08) * np.sin(
                self.rotation_factor * np.pi / 2) + 0.01
            palm_pose[2] = self.manipulated_object.get_pose().p[2] + 0.08

        elif self.object_category in ['assembling_kits']:
            palm_pose[0] += 0.05

        r = R.from_euler('xyz', [0, 180, 0], degrees=True)
        quat = r.as_quat()[[3, 0, 1, 2]]  #x,y,z,w
        target_pose = sapien.Pose(palm_pose, quat)
        result, success, error = self.pinocchio_model.compute_inverse_kinematics(
            self.palm_link.get_index(), target_pose, self.robot.get_qpos(),
            [1] * 6 + [0] * 6)

        result[6] = np.pi / 2 * 0

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
    env = WriteRLEnv(
        use_gui=True,
        robot_name="xarm7_gripper_down",
        object_name="any_train",  #101727
        object_category="pen",  #assembling_kits #02946921
        frame_skip=10,
        use_visual_obs=False,
        use_orientation=False,
        use_buffer=True,
        render_mesh=True,
        index=2)
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
    viewer.set_camera_rpy(r=0, p=0, y=np.pi)  # change the viewer direction
    # viewer.set_camera_xyz(-1, 0, 1)

    env.reset()

    # viewer.toggle_pause(True)
    pose = env.palm_link.get_pose()
    # while not viewer.closed:
    #     action = np.zeros(robot_dof)
    #     action[0] = 0.1
    #     obs, reward, done, _ = env.step(action)

    #     if i % 10 == 0:
    #         pose_error = pose.inv() * env.palm_link.get_pose()
    #         env.reset_env()
    #         print(pose_error)
    #     env.render()

    # while not viewer.closed:
    #     env.render()

    frame = 0
    while not viewer.closed:
        frame += 1
        action = np.zeros(robot_dof)

        action[7:] = 2
        obs, reward, done, _ = env.step(action)
        if frame % 200 == 0:
            env.reset()
            frame = 0

        #env.simple_step()
        env.render()


if __name__ == '__main__':
    main_env()