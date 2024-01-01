import argparse

import numpy as np
import sapien.core as sapien
from sapien.utils import Viewer

from abc import abstractmethod
from functools import cached_property
from operator import truediv
from typing import Dict, Optional, Callable, List, Union, Tuple

import gym
import numpy as np
import sapien.core as sapien
import transforms3d
import pdb
from hand_teleop.env.sim_env.base import BaseSimulationEnv
from hand_teleop.env.sim_env.constructor import add_default_scene_light
from hand_teleop.kinematics.kinematics_helper import PartialKinematicModel
from hand_teleop.utils.common_robot_utils import load_robot, generate_arm_robot_hand_info, \
    generate_free_robot_hand_info, FreeRobotInfo, ArmRobotInfo
from hand_teleop.utils.random_utils import np_random
import matplotlib.pyplot as plt
from hand_teleop.utils.common_robot_utils import load_robot, LPFilter
from hand_teleop.real_world import lab
from scipy.spatial.transform import Rotation as R

VISUAL_OBS_RETURN_TORCH = False
MAX_DEPTH_RANGE = 2.5
gl2sapien = sapien.Pose(q=np.array([0.5, 0.5, -0.5, -0.5]))
from hand_teleop.utils.camera_utils import fetch_texture, generate_imagination_pc_from_obs


def add_gripper_constraint(robot, scene):
    # base = [l for l in robot.get_links() if l.name == "robotiq_arg2f_base_link"][0]
    # lp = [l for l in robot.get_links() if l.name == "left_inner_finger_pad"][0]
    # rp = [l for l in robot.get_links() if l.name == "right_inner_finger_pad"][0]

    # lif = [l for l in robot.get_links() if l.name == "left_inner_knuckle"][0]
    # lok = [l for l in robot.get_links() if l.name == "left_outer_knuckle"][0]
    # rif = [l for l in robot.get_links() if l.name == "right_inner_knuckle"][0]
    # rok = [l for l in robot.get_links() if l.name == "right_outer_knuckle"][0]

    # rd = scene.create_drive(base, sapien.Pose(), rp, sapien.Pose())
    # rd.lock_motion(0, 0, 0, 1, 1, 1)
    # ld = scene.create_drive(base, sapien.Pose(), lp, sapien.Pose(q=np.array([0, 0, 0, 1])))
    # ld.lock_motion(0, 0, 0, 1, 1, 1)
    # ld2 = scene.create_drive(lif, sapien.Pose(), lok, sapien.Pose())
    # ld2.lock_motion(0, 0, 0, 1, 1, 1)
    # rd2 = scene.create_drive(rif, sapien.Pose(), rok, sapien.Pose())
    # rd2.lock_motion(0, 0, 0, 1, 1, 1)

    outer_knuckle = next(j for j in robot.get_active_joints()
                         if j.name == "right_outer_knuckle_joint")
    outer_finger = next(j for j in robot.get_active_joints()
                        if j.name == "right_inner_finger_joint")
    inner_knuckle = next(j for j in robot.get_active_joints()
                         if j.name == "right_inner_knuckle_joint")

    pad = outer_finger.get_child_link()
    lif = inner_knuckle.get_child_link()

    T_pw = pad.pose.inv().to_transformation_matrix()
    p_w = (outer_finger.get_global_pose().p +
           inner_knuckle.get_global_pose().p -
           outer_knuckle.get_global_pose().p)
    T_fw = lif.pose.inv().to_transformation_matrix()
    p_f = T_fw[:3, :3] @ p_w + T_fw[:3, 3]
    p_p = T_pw[:3, :3] @ p_w + T_pw[:3, 3]

    right_drive = scene.create_drive(lif, sapien.Pose(p_f), pad,
                                     sapien.Pose(p_p))
    right_drive.lock_motion(1, 1, 1, 0, 0, 0)

    outer_knuckle = next(j for j in robot.get_active_joints()
                         if j.name == "left_outer_knuckle_joint")
    outer_finger = next(j for j in robot.get_active_joints()
                        if j.name == "left_inner_finger_joint")
    inner_knuckle = next(j for j in robot.get_active_joints()
                         if j.name == "left_inner_knuckle_joint")

    pad = outer_finger.get_child_link()
    lif = inner_knuckle.get_child_link()

    T_pw = pad.pose.inv().to_transformation_matrix()
    p_w = (outer_finger.get_global_pose().p +
           inner_knuckle.get_global_pose().p -
           outer_knuckle.get_global_pose().p)
    T_fw = lif.pose.inv().to_transformation_matrix()
    p_f = T_fw[:3, :3] @ p_w + T_fw[:3, 3]
    p_p = T_pw[:3, :3] @ p_w + T_pw[:3, 3]

    left_drive = scene.create_drive(lif, sapien.Pose(p_f), pad,
                                    sapien.Pose(p_p))
    left_drive.lock_motion(1, 1, 1, 0, 0, 0)


def compute_inverse_kinematics(delta_pose_world, palm_jacobian, damping=0.05):
    lmbda = np.eye(6) * (damping**2)
    # When you need the pinv for matrix multiplication, always use np.linalg.solve but not np.linalg.pinv
    delta_qpos = palm_jacobian.T @ \
                 np.linalg.lstsq(palm_jacobian.dot(palm_jacobian.T) + lmbda, delta_pose_world, rcond=None)[0]

    return delta_qpos


def recover_action(action, limit):

    action = (action + 1) / 2 * (limit[:, 1] - limit[:, 0]) + limit[:, 0]

    return action


def recover_gripper_action(action, limit, init_action=None):

    if init_action is not None:

        action = (init_action - action) / 2 * (limit[:, 1] -
                                               limit[:, 0]) + limit[:, 0]

    else:
        action = (action) / 2 * (limit[:, 1] - limit[:, 0]) + limit[:, 0]

    return action


from hand_teleop.env.sim_env.base import BaseSimulationEnv
import gym


class RobotEnv(BaseSimulationEnv, gym.Env):

    def __init__(self,
                 use_gui=True,
                 frame_skip=5,
                 use_visual_obs=False,
                 regrasp=False,
                 use_filter=False,
                 noise_table=False,
                 **renderer_kwargs):

        super().__init__(use_gui=use_gui,
                         frame_skip=frame_skip,
                         use_visual_obs=use_visual_obs,
                         use_filter=use_filter,
                         **renderer_kwargs)
        self.data_collection = False
        self.noise_table = noise_table
        self.camera_infos: Dict[str, Dict] = {}
        self.camera_pose_noise: Dict[str, Tuple[
            Optional[float], sapien.Pose]] = {
            }  # tuple for noise level and original camera pose
        self.imagination_infos: Dict[str, float] = {}
        self.imagination_data: Dict[str, Dict[str,
                                              Tuple[sapien.ActorBase,
                                                    np.ndarray, int]]] = {}
        self.imaginations: Dict[str, np.ndarray] = {}

        # RL related attributes
        self.is_robot_free: Optional[bool] = None
        self.arm_dof: Optional[int] = None
        self.rl_step: Optional[Callable] = None
        self.get_observation: Optional[Callable] = None
        self.robot_collision_links: Optional[List[sapien.Actor]] = None
        self.robot_info: Optional[Union[ArmRobotInfo, FreeRobotInfo]] = None
        self.velocity_limit: Optional[np.ndarray] = None
        self.kinematic_model: Optional[PartialKinematicModel] = None

        # Robot cache
        self.control_time_step = None
        self.ee_link_name = None
        self.ee_link: Optional[sapien.Actor] = None
        self.cartesian_error = None

        self.render_tactile = False
        self.task_index = None

        self.pre_sum = 0

        self.init_action = None

    def setup(self, robot_name):

        info = generate_arm_robot_hand_info()[robot_name]
        self.robot_info = info

        #================================== load robot ==================================

        self.robot = load_robot(self.scene,
                                robot_name,
                                disable_self_collision=True)
        self.pinocchio_model = self.robot.create_pinocchio_model()

        init_pos = np.array(lab.ROBOT2BASE.p) + self.robot_info.root_offset
        init_pose = sapien.Pose(init_pos,
                                transforms3d.euler.euler2quat(0, 0, 0))
        self.robot.set_pose(init_pose)
        #================================== load robot ==================================

        #================================== gripper constraint ==================================
        add_gripper_constraint(self.robot, self.scene)

        right_joint = next(j for j in self.robot.get_active_joints()
                           if j.name == "right_outer_knuckle_joint")
        right_joint.set_drive_property(1e5, 2000, 0.1)
        # right_joint.set_drive_target(0.2)

        left_joint = next(j for j in self.robot.get_active_joints()
                          if j.name == "left_outer_knuckle_joint")
        left_joint.set_drive_property(1e5, 2000, 0.1)
        # left_joint.set_drive_target(0.2)

        if self.task_index in [7, 11]:
            gripper_joints = self.robot.get_active_joints()[-6:]
            for joints in gripper_joints:
                joints.set_limits([[0.60, joints.get_limits()[0, 1]]])
        elif self.task_index in [8, 10]:
            gripper_joints = self.robot.get_active_joints()[-6:]
            for joints in gripper_joints:
                joints.set_limits([[0.65, joints.get_limits()[0, 1]]])
        elif self.task_index == 9:
            gripper_joints = self.robot.get_active_joints()[-6:]
            for joints in gripper_joints:
                joints.set_limits([[0.75, joints.get_limits()[0, 1]]])
        elif self.task_index == 0:
            gripper_joints = self.robot.get_active_joints()[-6:]
            for joints in gripper_joints:
                joints.set_limits([[0.35, joints.get_limits()[0, 1]]])
        else:
            gripper_joints = self.robot.get_active_joints()[-6:]
            for joints in gripper_joints:
                joints.set_limits([[0.20, joints.get_limits()[0, 1]]])

        #================================== gripper constraint ==================================

        self.arm_dof = 7

        #================================== KinematicModel ==================================
        from hand_teleop.kinematics.kinematics_helper import PartialKinematicModel

        start_joint_name = self.robot.get_joints()[1].get_name()
        end_joint_name = self.robot.get_active_joints()[self.arm_dof -
                                                        1].get_name()
        self.kinematic_model = PartialKinematicModel(self.robot,
                                                     start_joint_name,
                                                     end_joint_name)
        ee_link_name = self.kinematic_model.end_link_name
        self.ee_link = [
            link for link in self.robot.get_links()
            if link.get_name() == ee_link_name
        ][0]
        #================================== KinematicModel ==================================

        #================================== other setting ==================================

        velocity_limit = np.array([0.2] * 5 + [2 * np.pi] * 1 + [1] +
                                  [1.0] * 6)  #slow down

        self.velocity_limit = np.stack([-velocity_limit, velocity_limit],
                                       axis=1)

        self.right_joint = next(j for j in self.robot.get_active_joints()
                                if j.name == "right_outer_knuckle_joint")
        # self.right_joint.set_drive_property(1e5, 2000, 0.1)
        self.right_joint.set_drive_target(0.0)

        self.left_joint = next(j for j in self.robot.get_active_joints()
                               if j.name == "left_outer_knuckle_joint")
        # self.left_joint.set_drive_property(1e5, 2000, 0.1)
        self.left_joint.set_drive_target(0)
        self.frame_skip = 10

        self.control_time_step = self.scene.get_timestep() * self.frame_skip

        self.robot_collision_links = [
            link for link in self.robot.get_links()
            if len(link.get_collision_shapes()) > 0
        ]
        #================================== other setting ==================================

        if self.is_robot_free:
            self.rl_step = self.free_sim_step
        else:
            self.rl_step = self.arm_sim_step

        # if self.task_index in [10, 11]:
        #     self.rl_step = self.arm_ref_traj_sim_step

        # Scene light and obs

        if self.use_visual_obs:

            if self.use_tactile:

                self.get_observation = self.get_TactilePC_observation
            else:
                self.get_observation = self.get_visual_observation

            if not self.no_rgb:
                add_default_scene_light(self.scene, self.renderer)
        else:

            if self.use_tactile:

                self.get_observation = self.get_TactileState_observation

            else:
                self.get_observation = self.get_oracle_state

    def get_info(self):

        return {}

    def update_cached_state(self):
        return

    @abstractmethod
    def is_done(self):
        pass

    def get_TactileState_observation(self):
        '''
        obtain tactile state and state for the link
        '''
        obs_dict = {}
        # obs_dict["oracle_state"] = self.get_oracle_state()
        obs_dict["oracle_state"] = self.get_oracle_state()
        obs_dict["tactile_image"] = self.get_tactile_state()
        if self.data_collection:

            obs_dict["angle"] = self.current_angles / 180 * np.pi

        return obs_dict

    def step(self, action: np.ndarray):

        self.rl_step(action)
        self.update_cached_state()

        self.update_imagination(reset_goal=False)
        obs = self.get_observation()

        reward = self.get_reward(action)
        done = self.is_done()
        info = self.get_info()

        if self.task_index == 0:
            info["success_index"] = self.target_obj_height_dist
            info["rewards"] = self.overall_rewards
            self.pre_sum = self.overall_rewards

        if self.eval:
            info["rewards"] = self.overall_rewards
            self.pre_sum = self.overall_rewards

            info["sum_angles"] = self.theta
            info["contact_angles"] = self.error_percentage

            if np.sum(self.robot_object_contact[:2]) >= 1:
                info["done"] = True
            else:
                info["done"] = False

        if self.data_collection:
            info["angle"] = self.current_angles / 180 * np.pi

        # Reference: https://github.com/openai/gym/blob/master/gym/wrappers/time_limit.py
        # Need to consider that is_done and timelimit can happen at the same time

        if self.current_step >= self.horizon:
            info["TimeLimit.truncated"] = not done
            done = True

        return obs, reward, done, info

    def update_imagination(self, reset_goal=False):

        for img_type, img_config in self.imagination_data.items():
            if img_type == "goal":
                if reset_goal:
                    imagination_goal = []
                    for link_name, (attr_name, points,
                                    img_class) in img_config.items():
                        pose = self.robot.get_pose().inv(
                        ) * self.__dict__[attr_name].get_pose()
                        # pose = self.robot.get_pose().inv(
                        # ) * self.manipulated_object.get_links()[-1].get_pose()
                        mat = pose.to_transformation_matrix()
                        transformed_points = points @ mat[:3, :
                                                          3].T + mat[:3, 3][
                                                              None, :]
                        imagination_goal.append(transformed_points)
                    self.imaginations["imagination_goal"] = np.concatenate(
                        imagination_goal, axis=0)

            if img_type == "robot":
                imagination_robot = []
                for link_name, (actor, points,
                                img_class) in img_config.items():
                    pose = self.robot.get_pose().inv() * actor.get_pose()
                    mat = pose.to_transformation_matrix()
                    transformed_points = points @ mat[:3, :3].T + mat[:3, 3][
                        None, :]
                    imagination_robot.append(transformed_points)
                    self.imaginations["imagination_robot"] = np.concatenate(
                        imagination_robot, axis=0)

    def reset_internal(self):
        self.current_step = 0
        if self.init_state is not None:
            self.scene.unpack(self.init_state)

        self.reset_env()
        if self.init_state is None:
            self.init_state = self.scene.pack()

        # Reset camera pose
        for cam_name, (noise_level,
                       original_pose) in self.camera_pose_noise.items():
            if noise_level is None:
                continue
            pos_noise = self.np_random.randn(3) * noise_level * 0.03
            rot_noise = self.np_random.randn(3) * noise_level * 0.1
            quat_noise = transforms3d.euler.euler2quat(*rot_noise)
            perturb_pose = sapien.Pose(pos_noise, quat_noise)
            self.cameras[cam_name].set_local_pose(original_pose * perturb_pose)

    def arm_ref_traj_sim_step(self, action: np.ndarray):

        current_qpos = self.robot.get_qpos()
        ee_link_last_pose = self.ee_link.get_pose()

        hand_limit = self.robot.get_qlimits()[self.arm_dof:]
        hand_qpos = recover_gripper_action(
            action[self.arm_dof:],
            hand_limit,
            self.init_action,
        )

        self.right_joint.set_drive_target(hand_qpos[0])
        # self.right_joint.set_drive_velocity_target(hand_qpos[0])
        self.left_joint.set_drive_target(hand_qpos[0])
        # self.right_joint.set_drive_velocity_target(hand_qpos[0])

        theta = self.current_step / 200 * np.pi * 2
        # theta = 0
        x = np.cos(theta)
        y = np.sin(theta)

        palm_pose = self.manipulated_object.get_pose().p

        palm_pose[0] = 0.55 + x * 0.1
        palm_pose[1] = y * 0.1
        palm_pose[2] = self.object_height

        if self.object_category in ["pen"]:

            palm_pose[2] = self.object_height + 0.05

        elif self.object_category in ["spoon"]:

            palm_pose[2] = self.object_height + 0.05

        elif self.object_category in ["kitchen"]:

            palm_pose[2] = self.manipulated_object.get_pose().p[2] + 0.08

        elif self.object_category in ['assembling_kits']:
            palm_pose[0] += 0.05

        r = R.from_euler('xyz', [0, 180, 0], degrees=True)
        quat = r.as_quat()[[3, 0, 1, 2]]  #x,y,z,w
        target_pose = sapien.Pose(palm_pose, quat)
        target_qpos, success, error = self.pinocchio_model.compute_inverse_kinematics(
            self.palm_link.get_index(), target_pose, self.robot.get_qpos(),
            [1] * 6 + [0] * 6)

        target_qpos[6] = np.pi / 2 * 0

        for i in range(self.arm_dof):
            self.robot.get_active_joints()[i].set_drive_target(target_qpos[i])
            self.robot.get_active_joints()[i].set_drive_velocity_target(0.01)

        for i in range(self.frame_skip):
            self.robot.set_qf(
                self.robot.compute_passive_force(
                    external=False, coriolis_and_centrifugal=False))
            self.scene.step()
        self.current_step += 1

        ee_link_new_pose = self.ee_link.get_pose()
        relative_pos = ee_link_new_pose.p - ee_link_last_pose.p

        self.cartesian_error = np.linalg.norm(relative_pos -
                                              np.array(palm_pose))

    def arm_sim_step(self, action: np.ndarray):

        self.arm_dof = 7
        current_qpos = self.robot.get_qpos()
        ee_link_last_pose = self.ee_link.get_pose()
        action = np.clip(action, -1, 1)

        # action[2] = np.clip(action[2], 0.12, 10000)
        if self.task_index == 0:  #insertion
            action[[3, 4, 5]] = 0
        elif self.task_index == 1 or self.task_index == 5:
            action[[0, 1, 3, 4]] = 0
        elif self.task_index in [6, 7, 8, 9, 11]:
            robot_action = np.zeros(6)

            robot_action[[
                0,
                2,
                4,
            ]] = action
            action = robot_action
            # action[[1, 3, 5]] = 0

        target_root_velocity = recover_action(action[:6],
                                              self.velocity_limit[:6])

        palm_jacobian = self.kinematic_model.compute_end_link_spatial_jacobian(
            current_qpos[:self.arm_dof])

        arm_qvel = compute_inverse_kinematics(target_root_velocity,
                                              palm_jacobian)[:self.arm_dof]
        arm_qvel = np.clip(arm_qvel, -np.pi / 1, np.pi / 1)
        if self.task_index in [9]:
            ids = [0, 2]
            arm_qvel[ids] = 0

        arm_qpos = arm_qvel * self.control_time_step + self.robot.get_qpos(
        )[:self.arm_dof]
        if self.task_index in [9]:
            arm_qpos[ids] = 0
        #==================================strict contact constraints ====================================
        hand_limit = self.robot.get_qlimits()[self.arm_dof:]

        # if self.task_index == 7:
        #     hand_limit[:, 0] = 0.2

        # hand_qpos = recover_gripper_action(
        #     action[-1],
        #     hand_limit,
        #     self.init_action,
        # )

        # target_qpos = np.concatenate([arm_qpos, hand_qpos])
        target_qpos = arm_qpos

        target_qvel = np.zeros_like(target_qpos)
        target_qvel[:self.arm_dof] = arm_qvel

        # self.robot.set_drive_target(target_qpos)
        # self.robot.set_drive_velocity_target(target_qvel)

        for i in range(self.arm_dof):
            self.robot.get_active_joints()[i].set_drive_target(target_qpos[i])
            self.robot.get_active_joints()[i].set_drive_velocity_target(
                target_qvel[i])

        # self.right_joint.set_drive_target(hand_qpos[0])
        self.right_joint.set_drive_target(0.85)
        # self.left_joint.set_drive_target(hand_qpos[0])
        self.left_joint.set_drive_target(0.85)

        for i in range(self.frame_skip):
            self.robot.set_qf(
                self.robot.compute_passive_force(
                    external=False, coriolis_and_centrifugal=False))
            self.scene.step()
        self.current_step += 1

        ee_link_new_pose = self.ee_link.get_pose()
        relative_pos = ee_link_new_pose.p - ee_link_last_pose.p
        self.cartesian_error = np.linalg.norm(relative_pos -
                                              target_root_velocity[:3] *
                                              self.control_time_step)

        return target_root_velocity

        # r = R.from_quat(ee_link_last_pose.q[[1, 2, 3, 0]])
        # last_angles = r.as_euler('xyz', degrees=False)
        # r = R.from_quat(ee_link_new_pose.q[[1, 2, 3, 0]])
        # new_angles = r.as_euler('xyz', degrees=False)
        # print(new_angles - last_angles,
        #       target_root_velocity[3:] * self.control_time_step)


#=====================tactile===========================

    def setup_tactile_obs_config(self,
                                 config,
                                 use_expert=False,
                                 dagger_tactile=False):
        '''
        set up the tactile observation config
        '''

        self.render_tactile = True
        self.use_expert = use_expert
        self.dagger_tactile = dagger_tactile
        self.tactile_modality_name = config[
            "modality"]  # the name for the tactile
        self.tactile_modality_state_dim = config[
            "state_dimension"]  # the link state dimension for the tactile
        self.tactile_modality_dim = config["image_dimension"]

    #=====================tactile===========================

    def setup_visual_obs_config(self, config: Dict[str, Dict]):
        for name, camera_cfg in config.items():

            if name not in self.cameras.keys():
                raise ValueError(
                    f"Camera {name} not created. Existing {len(self.cameras)} cameras: {self.cameras.keys()}"
                )
            self.camera_infos[name] = {}
            banned_modality_set = {"point_cloud", "depth"}
            if len(banned_modality_set.intersection(set(
                    camera_cfg.keys()))) == len(banned_modality_set):
                raise RuntimeError(
                    f"Request both point_cloud and depth for same camera is not allowed. "
                    f"Point cloud contains all information required by the depth."
                )

            # Add perturb for camera pose
            cam = self.cameras[name]
            if "pose_perturb_level" in camera_cfg:
                cam_pose_perturb = camera_cfg.pop("pose_perturb_level")
            else:
                cam_pose_perturb = None
            self.camera_pose_noise[name] = (cam_pose_perturb, cam.get_pose())

            for modality, cfg in camera_cfg.items():
                if modality == "point_cloud":
                    if "process_fn" not in cfg or "num_points" not in cfg:
                        raise RuntimeError(
                            f"Missing process_fn or num_points in camera {name} point_cloud config."
                        )

                self.camera_infos[name][modality] = cfg

        modality = []
        for camera_cfg in config.values():
            modality.extend(camera_cfg.keys())
        modality_set = set(modality)
        if "rgb" in modality_set and self.no_rgb:
            raise RuntimeError(
                f"Only point cloud, depth, and segmentation are allowed when no_rgb is enabled."
            )

    def setup_imagination_config(self, config: Dict[str, Dict[str, int]]):
        from hand_teleop.utils.render_scene_utils import actor_to_open3d_mesh
        import open3d as o3d
        # Imagination can only be used with point cloud representation
        for name, camera_cfg in self.camera_infos.items():
            assert "point_cloud" in camera_cfg

        acceptable_imagination = ["robot", "goal", "contact"]
        # Imagination class: 0 (observed), 1 (robot), 2 (goal), 3 (contact)
        img_dict = {}

        collision_link_names = [
            link.get_name() for link in self.robot_collision_links
        ]
        for img_type, link_config in config.items():
            if img_type not in acceptable_imagination:
                raise ValueError(
                    f"Unknown Imagination config name: {img_type}.")
            if img_type == "robot":
                img_dict["robot"] = {}
                for link_name, point_size in link_config.items():
                    if link_name not in collision_link_names:
                        raise ValueError(
                            f"Link name {link_name} does not have collision geometry."
                        )
                    link = [
                        link for link in self.robot_collision_links
                        if link.get_name() == link_name
                    ][0]
                    o3d_mesh = actor_to_open3d_mesh(link,
                                                    use_collision_mesh=False,
                                                    use_actor_pose=False)
                    sampled_cloud = o3d_mesh.sample_points_uniformly(
                        number_of_points=point_size)
                    cloud_points = np.asarray(sampled_cloud.points)
                    img_dict["robot"][link_name] = (link, cloud_points, 1)
            elif img_type == "goal":
                img_dict["goal"] = {}
                # We do not use goal actor pointer to index pose. During reset, the goal actor may be removed.
                # Thus use goal actor here to fetch pose will lead to segfault
                # Instead, the attribute name is saved, so we can always find the latest goal actor
                for attr_name, point_size in link_config.items():
                    goal_sphere = o3d.geometry.TriangleMesh.create_sphere(
                        radius=0.05)
                    sampled_cloud = goal_sphere.sample_points_uniformly(
                        number_of_points=point_size)
                    # link = self.manipulated_object.get_links()[-1]
                    # o3d_mesh = actor_to_open3d_mesh(link,
                    #                                 use_collision_mesh=False,
                    #                                 use_actor_pose=False)
                    # sampled_cloud = o3d_mesh.sample_points_uniformly(
                    #     number_of_points=point_size)
                    cloud_points = np.asarray(sampled_cloud.points)
                    img_dict["goal"][attr_name] = (attr_name, cloud_points, 2)
            else:
                raise NotImplementedError

        self.imagination_infos = config
        self.imagination_data = img_dict

    def update_imagination(self, reset_goal=False):

        for img_type, img_config in self.imagination_data.items():
            if img_type == "goal":
                if reset_goal:
                    imagination_goal = []
                    for link_name, (attr_name, points,
                                    img_class) in img_config.items():
                        pose = self.robot.get_pose().inv(
                        ) * self.__dict__[attr_name].get_pose()
                        # pose = self.robot.get_pose().inv(
                        # ) * self.manipulated_object.get_links()[-1].get_pose()
                        mat = pose.to_transformation_matrix()
                        transformed_points = points @ mat[:3, :
                                                          3].T + mat[:3, 3][
                                                              None, :]
                        imagination_goal.append(transformed_points)
                    self.imaginations["imagination_goal"] = np.concatenate(
                        imagination_goal, axis=0)

            if img_type == "robot":
                imagination_robot = []
                for link_name, (actor, points,
                                img_class) in img_config.items():
                    pose = self.robot.get_pose().inv() * actor.get_pose()
                    mat = pose.to_transformation_matrix()
                    transformed_points = points @ mat[:3, :3].T + mat[:3, 3][
                        None, :]
                    imagination_robot.append(transformed_points)
                    self.imaginations["imagination_robot"] = np.concatenate(
                        imagination_robot, axis=0)

    def get_robot_state(self):
        raise NotImplementedError

    def get_oracle_state(self):
        raise NotImplementedError

    def get_visual_observation(self):
        camera_obs = self.get_camera_obs()
        robot_obs = self.get_robot_state()
        oracle_obs = self.get_oracle_state()
        camera_obs.update(dict(state=robot_obs, oracle_state=oracle_obs))
        return camera_obs

    #=====================tactile===========================
    def get_TactilePC_observation(self):
        camera_obs = self.get_camera_obs()
        robot_obs = self.get_robot_state()
        oracle_obs = self.get_oracle_state()
        tactile_state = self.get_tactile_state()

        camera_obs.update(dict(
            state=robot_obs,
            oracle_state=oracle_obs,
        ))
        camera_obs[self.tactile_modality_name] = tactile_state

        return camera_obs

    #=====================tactile===========================

    def get_camera_obs(self):
        self.scene.update_render()
        obs_dict = {}
        for name, camera_cfg in self.camera_infos.items():
            cam = self.cameras[name]

            modalities = list(camera_cfg.keys())
            texture_names = []

            for modality in modalities:
                if modality == "rgb":
                    texture_names.append("Color")
                elif modality == "depth":
                    texture_names.append("Position")
                elif modality == "point_cloud":
                    texture_names.append("Position")
                elif modality == "segmentation":
                    texture_names.append("Segmentation")
                else:
                    raise ValueError(
                        f"Visual modality {modality} not supported.")

            await_dl_list = cam.take_picture_and_get_dl_tensors_async(
                texture_names)
            dl_list = await_dl_list.wait()

            for i, modality in enumerate(modalities):

                key_name = f"{name}-{modality}"
                dl_tensor = dl_list[i]
                shape = sapien.dlpack.dl_shape(dl_tensor)

                if modality in ["segmentation"]:
                    # TODO: add uint8 async
                    import torch
                    output_array = torch.from_dlpack(dl_tensor).cpu().numpy()
                else:
                    output_array = np.zeros(shape, dtype=np.float32)

                    sapien.dlpack.dl_to_numpy_cuda_async_unchecked(
                        dl_tensor, output_array)
                    sapien.dlpack.dl_cuda_sync()
                if modality == "rgb":
                    obs = output_array[..., :3]
                elif modality == "depth":
                    obs = -output_array[..., 2:3]
                    obs[obs[..., 0] >
                        1.5] = 0  # Set depth out of range to be 0 #MAX_DEPTH_RANGE

                elif modality == "point_cloud":
                    obs = np.reshape(output_array[..., :3], (-1, 3))

                    camera_pose = self.get_camera_to_robot_pose(name)

                    kwargs = camera_cfg["point_cloud"].get(
                        "process_fn_kwargs", {})
                    obs = camera_cfg["point_cloud"]["process_fn"](
                        obs,
                        camera_pose,
                        camera_cfg["point_cloud"]["num_points"],
                        self.np_random,
                        floor_height=self.floor_height,
                        upper_bound=self.palm_link.get_pose().p[2],
                        noise_table=self.noise_table,
                        **kwargs)

                    if "additional_process_fn" in camera_cfg["point_cloud"]:
                        for fn in camera_cfg["point_cloud"][
                                "additional_process_fn"]:
                            obs = fn(obs, self.np_random)

                elif modality == "segmentation":
                    obs = output_array[..., :2].astype(np.uint8)
                else:
                    raise RuntimeError(
                        "What happen? you should not see this error!")

                obs_dict[key_name] = obs

        if len(self.imaginations) > 0:
            obs_dict.update(self.imaginations)

        return obs_dict

    def get_camera_to_robot_pose(self, camera_name):
        gl_pose = self.cameras[camera_name].get_pose()
        camera_pose = gl_pose * gl2sapien
        camera2robot = self.robot.get_pose().inv() * camera_pose
        return camera2robot.to_transformation_matrix()

    @property
    def action_dim(self):
        return 3

    @cached_property
    def action_space(self):
        print(self.task_index)
        return gym.spaces.Box(low=-1, high=1, shape=(self.action_dim, ))

    @cached_property
    def observation_space(self):
        high = np.inf * np.ones(self.obs_dim)
        low = -high
        state_space = gym.spaces.Box(low=low, high=high)

        if not self.use_visual_obs:

            #=======================================================
            #=====================tactile===========================
            if self.render_tactile and not self.use_expert:  # set the gym space for the  state and tactile render
                obs_dict = {}
                obs_dict[self.tactile_modality_name] = gym.spaces.Box(
                    low=0,
                    high=255,
                    dtype=np.uint8,
                    shape=self.tactile_modality_dim)

                state_dim = len(self.get_oracle_state())
                state_dim = gym.spaces.Box(low=-np.inf * np.ones(state_dim),
                                           high=np.inf * np.ones(state_dim))
                obs_dict["oracle_state"] = state_dim  #oracle state

                return gym.spaces.Dict(obs_dict)

            elif self.render_tactile and self.use_expert and self.dagger_tactile:  # set the gym space for the  state and tactile render
                obs_dict = {}
                obs_dict[self.tactile_modality_name] = gym.spaces.Box(
                    low=0,
                    high=255,
                    dtype=np.uint8,
                    shape=self.tactile_modality_dim)

                state_dim = len(self.get_oracle_state())
                state_dim = gym.spaces.Box(low=-np.inf * np.ones(state_dim),
                                           high=np.inf * np.ones(state_dim))
                obs_dict["oracle_state"] = state_dim  #oracle state

                return gym.spaces.Dict(obs_dict)

            elif self.render_tactile and self.use_expert and not self.dagger_tactile:
                return state_space

            else:
                return state_space
            #=====================tactile===========================
            #=======================================================
        else:
            oracle_dim = len(self.get_oracle_state())
            oracle_space = gym.spaces.Box(low=-np.inf * np.ones(oracle_dim),
                                          high=np.inf * np.ones(oracle_dim))
            obs_dict = {"state": state_space, "oracle_state": oracle_space}

            #=====================tactile===========================
            if self.render_tactile:  # tactile space

                obs_dict[self.tactile_modality_name] = gym.spaces.Box(
                    low=0,
                    high=255,
                    dtype=np.uint8,
                    shape=self.tactile_modality_dim)

                # obs_dict[self.tactile_modality_name] = gym.spaces.Box(
                #     low=-np.inf, high=np.inf, shape=self.tactile_modality_dim)
            #=====================tactile===========================
            for cam_name, cam_cfg in self.camera_infos.items():
                cam = self.cameras[cam_name]
                resolution = (cam.height, cam.width)
                for modality_name in cam_cfg.keys():
                    key_name = f"{cam_name}-{modality_name}"
                    if modality_name == "rgb":
                        spec = gym.spaces.Box(low=0,
                                              high=1,
                                              shape=resolution + (3, ))
                    elif modality_name == "depth":
                        spec = gym.spaces.Box(low=0,
                                              high=MAX_DEPTH_RANGE,
                                              shape=resolution + (1, ))
                    elif modality_name == "point_cloud":
                        spec = gym.spaces.Box(
                            low=-np.inf,
                            high=np.inf,
                            shape=(cam_cfg[modality_name]["num_points"], ) +
                            (3, ))
                    elif modality_name == "segmentation":
                        spec = gym.spaces.Box(low=0,
                                              high=255,
                                              shape=resolution + (2, ),
                                              dtype=np.uint8)
                    else:
                        raise RuntimeError(
                            "What happen? you should not see this error!")
                    obs_dict[key_name] = spec

            if len(self.imagination_infos) > 0:
                self.update_imagination(reset_goal=True)
                for img_name, points in self.imaginations.items():
                    num_points = points.shape[0]
                    obs_dict[img_name] = gym.spaces.Box(low=-np.inf,
                                                        high=np.inf,
                                                        shape=(num_points, 3))

            return gym.spaces.Dict(obs_dict)

    def setup_tactile_camera_from_config(self, conf):
        assert self.use_tactile == True

        near, far = conf['near'], conf['far']

        width, height = conf['width'], conf['height']

        self.link_cameras = []

        for link in self.finger_tip_links:

            camera = self.scene.add_mounted_camera(
                name=link.get_name(),
                actor=link,
                pose=sapien.Pose(
                    # q=[0.77, 0, 0, 0.77]
                ),  # relative to the mounted actor
                width=width,
                height=height,
                fovy=np.deg2rad(conf['fovy']),
                fovx=np.deg2rad(conf['fovx']),
                near=near,
                far=far,
            )
            # print("name:", link.get_name(), "link pose", link.get_pose(),
            #       "camera:", camera.get_pose())
            camera.set_local_pose(sapien.Pose(p=[-0.0, 0.0, 0.015]))
            self.link_cameras.append(camera)


def compute_inverse_kinematics(delta_pose_world, palm_jacobian, damping=0.05):
    lmbda = np.eye(6) * (damping**2)
    # When you need the pinv for matrix multiplication, always use np.linalg.solve but not np.linalg.pinv
    delta_qpos = palm_jacobian.T @ \
                 np.linalg.lstsq(palm_jacobian.dot(palm_jacobian.T) + lmbda, delta_pose_world, rcond=None)[0]

    return delta_qpos
