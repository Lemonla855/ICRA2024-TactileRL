from functools import cached_property
from typing import Optional, ClassVar

import numpy as np
import sapien.core as sapien
import transforms3d
from gym.utils import seeding
from sapien.utils import Viewer

from hand_teleop.env.rl_env.base import BaseRLEnv

from hand_teleop.env.sim_env.slip_env import LabSlipRelocateEnv
from hand_teleop.real_world import lab
#=====================tactile===========================
from hand_teleop.utils.tactile_utils import obtain_tactile_postion, obtain_tactile_force, Tactile
import pdb
# import torch
# from hand_teleop.utils.sapien_render import fetch_tactile_image
#=====================tactile===========================
OBJECT_LIFT_LOWER_LIMIT = -0.03


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


class LabSlipRelocateRLEnv(LabSlipRelocateEnv, BaseRLEnv):

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
                 use_tips=True,
                 use_buffer=False,
                 reduced_state=False,
                 **renderer_kwargs):
        if "allegro" not in robot_name or "free" in robot_name:
            raise ValueError(
                f"Robot name: {robot_name} is not valid xarm allegro robot.")

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
        #     "link_3.0_tip_fsr", "link_2.0_fsr", "link_1.0_fsr",
        #     "link_7.0_tip_fsr", "link_6.0_fsr", "link_5.0_fsr",
        #     "link_11.0_tip_fsr", "link_10.0_fsr", "link_9.0_fsr"
        # ]

        finger_tip_names = [
            "link_15.0_tip_fsr", "link_3.0_tip_fsr", "link_7.0_tip_fsr",
            "link_11.0_tip_fsr"
        ]

        # if "bottle" in self.object_category:

        #     # finger_contact_link_name = [
        #     #     "link_15.0_tip_fsr", "link_3.0_tip_fsr", "link_7.0_tip_fsr",
        #     #     "link_11.0_tip_fsr", "link_0.0_fsr", "link_4.0_fsr",
        #     #     "link_8.0_fsr"
        #     # ]
        #     finger_contact_link_name = [
        #         "link_15.0_tip_fsr", "link_15.0_fsr", "link_14.0_fsr",
        #         "link_3.0_tip_fsr", "link_2.0_fsr", "link_1.0_fsr",
        #         "link_7.0_tip_fsr", "link_6.0_fsr", "link_5.0_fsr",
        #         "link_11.0_tip_fsr", "link_10.0_fsr", "link_9.0_fsr",
        #         "link_0.0_fsr", "link_4.0_fsr", "link_8.0_fsr"
        #     ]
        # else:

        finger_contact_link_name = [
            "link_15.0_tip", "link_15.0_tip_fsr", "link_15.0_fsr",
            "link_14.0_fsr", "link_3.0_tip", "link_3.0_tip_fsr",
            "link_2.0_fsr", "link_1.0_fsr", "link_7.0_tip", "link_7.0_tip_fsr",
            "link_6.0_fsr", "link_5.0_fsr", "link_11.0_tip",
            "link_11.0_tip_fsr", "link_10.0_fsr", "link_9.0_fsr",
            "link_0.0_fsr", "link_4.0_fsr", "link_8.0_fsr"
        ]

       
        #==========================contact =================================
        finger_box_names = [
            "link_15.0_tip", "link_3.0_tip", "link_7.0_tip", "link_11.0_tip"
        ]
        #==========================contact =================================
        #=====================tactile===========================
        '''
        TODO:Check whether need to modify
        '''
        # finger_contact_link_name = [
        #     "link_15.0_tip", "link_15.0_box", "link_15.0", "link_14.0",
        #     "link_3.0_tip", "link_3.0_box", "link_3.0", "link_2.0", "link_1.0",
        #     "link_7.0_tip", "link_7.0_box", "link_7.0", "link_6.0", "link_5.0",
        #     "link_11.0_tip", "link_11.0_box", "link_11.0", "link_10.0",
        #     "link_9.0"
        # ]
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
        # self.finger_contact_ids = np.array([0] * 4 + [1] * 5 + [2] * 5 +
        #                                    [3] * 5 + [4])
        #=====================tactile===========================

        #==============================contact============================
        # self.finger_contact_ids = np.array([0] * 4 + [1] * 4 + [2] * 4 +
        #                                    [3] * 4 + [4] * 3)
        self.finger_contact_ids = np.array([0] * 1 + [1] * 1 + [2] * 1 +
                                           [3] * 1 + [4] * 3)

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
        # self.create_collision([0, 0.15, 0.04], [0.707, 0, 0, 0.707])
        # if "lighter" in self.object_category:

        self.collision1 = None

        #==============================contact============================

    def create_collision(self, pos, ori):
        builder = self.scene.create_actor_builder()

        obj_pose = sapien.Pose(np.array(pos), np.array(ori))
        obj_material = self.scene.create_physical_material(1, 0.5, 0.01)

        obj_half_size = [0.002, 0.4, pos[2] * 2]

        # builder.add_box_collision(pose=top_pose,
        #                           half_size=table_half_size,
        #                           material=top_material)
        # builder.add_box_visual(
        #     pose=top_pose,
        #     half_size=table_half_size,
        # )
        # collision = builder.build_static("collision")

        builder.add_multiple_collisions_from_file(str("box.stl"),
                                                  scale=obj_half_size,
                                                  material=obj_material,
                                                  pose=obj_pose)
        builder.add_visual_from_file(str("box.stl"),
                                     scale=obj_half_size,
                                     pose=obj_pose)
        collision = builder.build_static("collision")
        return collision

    def update_cached_state(self):

        for i, link in enumerate(self.finger_tip_links):
            self.finger_tip_pos[i] = self.finger_tip_links[i].get_pose().p
        check_contact_links = self.finger_contact_links

        if "bottle" in self.object_category:
            self.contact_boolean, _ = self.check_actor_pair_contacts(
                check_contact_links,
                self.manipulated_object.get_links()[-1])

        else:
            self.contact_boolean, _ = self.check_actor_pair_contacts(
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

            if self.reduced_state:
                return np.concatenate([
                    robot_qpos_vec, self.palm_pos_in_base,
                    self.robot_object_contact,
                    self.init_obj_pose - self.base_frame_pos,
                    self.target_pose.p - self.base_frame_pos,
                    self.target_pose.q
                ])

            else:
                return np.concatenate([
                    robot_qpos_vec, self.palm_pos_in_base,
                    self.init_obj_pose - self.base_frame_pos,
                    self.target_pose.p - self.base_frame_pos,
                    self.target_pose.q, self.contact_boolean
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
            # return np.concatenate([
            #     robot_qpos_vec, self.palm_pos_in_base,
            #     self.init_obj_pose - self.base_frame_pos,
            #     self.target_pose.p - self.base_frame_pos, self.target_pose.q
            # ])

    #=====================tactile===========================
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

    def angle_diff(self, quat_a, quat_b):
        # Subtract quaternions and extract angle between them.
        quat_diff = quat_mul(quat_a, quat_conjugate(quat_b))
        from scipy.spatial.transform import Rotation as R
        r = R.from_quat([quat_diff[[1, 2, 3, 0]]])
        # print(abs(r.as_euler('xyz', degrees=False)[0, 0]))
        a_diff = 2 * np.arccos(np.clip(quat_diff[..., 0], -1., 1.))
        return a_diff, abs(r.as_euler('xyz', degrees=False)[0, 0])

    def get_reward(self, action):
        finger_object_dist = np.linalg.norm(self.object_in_tip,
                                            axis=1,
                                            keepdims=False)
        finger_object_dist = np.clip(finger_object_dist, 0.03, 0.8)
        reward = np.sum(1.0 / (0.06 + finger_object_dist) *
                        self.finger_reward_scale)

        # at least one tip and palm or two tips are contacting obj. Thumb contact is required.
        is_contact = np.sum(self.robot_object_contact) >= 2

        # if "bottle" in self.object_category:
        #     is_contact = np.sum(self.robot_object_contact[:2]) >= 2

        # wall_contact, _ = self.check_actor_pair_contacts(
        #     [self.collision],
        #     self.manipulated_object.get_links()[1])

        diff_angle, z_diff = self.angle_diff(self.object_pose.q,
                                             self.target_pose.q)

        if is_contact:
            reward += 0.5
            lift = np.clip(self.object_lift, 0, 0.2)
            reward += 30 * lift
            if lift > 0.02:
                reward += 1
            # reward += 2.0 / (0.15 - self.object_pose.p[1] + 0.04)``
            # if diff_angle <1.0 :
            # if wall_contact:
            # #     reward += 1
            target_obj_dist = np.linalg.norm(self.target_in_object[2])
            reward += 1.0 / (0.04 + target_obj_dist)

            # if target_obj_dist < 0.1:

            # theta = self.target_in_object_angle[0]
            # reward += 4.0 / (0.4 + z_diff) * self.rotation_reward_weight * 3

            reward += (self.pre_diff_angle - z_diff) * 5000

        action_penalty = np.sum(np.clip(self.robot.get_qvel(), -1, 1)**
                                2) * -0.01
        controller_penalty = (self.cartesian_error**2) * -1e3
        self.pre_diff_angle = z_diff

        return (reward + action_penalty + controller_penalty) / 10 + np.sum(
            self.robot_object_contact *
            [2, 1, 0, 0, -3]) / 6  #- palm_contact_boolean - np.sum(
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

        # Set robot pose
        init_pos = np.array(lab.ROBOT2BASE.p) + self.robot_info.root_offset
        init_pose = sapien.Pose(init_pos,
                                transforms3d.euler.euler2quat(0, 0, 0))
        self.robot.set_pose(init_pose)

        self.diff_angle = 0
        self.pre_diff_angle = np.pi / 2

        pose = sapien.Pose([0, 0, 0.10], [0.707, -0.707, 0, 0])
        self.manipulated_object.set_pose(pose)

        for _ in range(2):

            self.step(np.zeros(self.robot.dof))

        self.object_height = self.manipulated_object.get_pose().p[2]
        self.init_obj_pose = self.manipulated_object.get_pose().p

        pose = sapien.Pose([0, 0, 0.10], self.manipulated_object_pose.q)
        self.manipulated_object.set_pose(pose)
        pose = self.generate_random_target_pose(self.randomness_scale)
        if self.target_object is not None:
            self.target_object.set_pose(pose)
        self.target_pose = pose

        qpos = self.reset_robot()
        self.robot.set_qpos(qpos)
        self.robot.set_drive_target(qpos)

        # if "lighter" in self.object_category:

        # if self.collision1 is None:

        #     self.collision1 = self.create_collision(
        #         [0.0, -self.object_height - 0.07, 0.07], [0.707, 0, 0, 0.707])
        #     self.collision2 = self.create_collision([-0.07, 0, 0.03],
        #                                             [1, 0, 0, 0])
        #     self.collision3 = self.create_collision([-0.05, -0.2, 0.04],
        #                                             [0.707, 0, 0, 0.707])
        #     self.collision4 = self.create_collision([-0.05, 0.2, 0.04],
        #                                             [0.707, 0, 0, 0.707])

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

            palm_pose = self.manipulated_object.get_links()[-1].get_pose().p

            palm_pose[0] += 0.4
            palm_pose[1] -= 0
            palm_pose[2] -= 0.0

            model = self.robot.create_pinocchio_model()

            from scipy.spatial.transform import Rotation as R
            r = R.from_euler('xyz', [90, 0, 0], degrees=True)
            quat = r.as_quat()[[3, 0, 1, 2]]  #x,y,z,w

        elif "down" in self.robot_name:
            # if "bottle" in self.object_category:
            palm_pose = self.manipulated_object.get_links()[-1].get_pose().p

            palm_pose[0] += 0.55
            palm_pose[1] += self.object_height / 2 - 0.04

            if self.object_category in ["lighter"]:
                palm_pose[2] += -0.03
            else:
                palm_pose[2] -= 0.04

            model = self.robot.create_pinocchio_model()
            from scipy.spatial.transform import Rotation as R
            # r = R.from_rotvec([np.pi / 2, np.pi / 2, 0])
            r = R.from_euler('xyz', [0, 90, -0], degrees=True)
            quat = r.as_quat()[[3, 0, 1, 2]]  #x,y,z,w
            # quat = [0.658, 0.196, 0.699, -0.201]
            # quat = [0.701, -0.704, 0.018, 0.022]

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
    env = LabSlipRelocateRLEnv(use_gui=True,
                               robot_name="xarm6_allegro_sfr_front",
                               object_name="4084",
                               object_category="bottle2",
                               frame_skip=10,
                               use_visual_obs=False,
                               use_orientation=False,
                               use_buffer=True)
    base_env = env
    robot_dof = env.robot.dof
    env.seed(0)

    tic = time()
    env.reset()

    tac = time()
    print(f"Reset time: {(tac - tic) * 1000} ms")

    # tic = time()
    # # for i in range(1000):
    # #     # action = np.random.rand(robot_dof) * 2 - 1
    # #     # action[2] = 0.1
    # #     obs, reward, done, _ = env.step(action)
    # tac = time()
    # print(f"Step time: {(tac - tic)} ms")

    viewer = Viewer(base_env.renderer)
    viewer.set_scene(base_env.scene)
    base_env.viewer = viewer
    viewer.set_camera_rpy(r=0, p=0, y=np.pi)  # change the viewer direction
    # viewer.set_camera_xyz(-1, 0, 1)

    # viewer.toggle_pause(True)
    pose = env.palm_link.get_pose()

    frame = 0
    while not viewer.closed:
        frame += 1
        # action = np.zeros(robot_dof)

        # # action[3:6] = 0
        # obs, reward, done, _ = env.step(action)

        if frame % 5000 == 0:
            env.reset()
            frame = 0

        #env.simple_step()
        env.render()


if __name__ == '__main__':
    main_env()
