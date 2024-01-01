import pdb
import numpy as np
import sapien.core as sapien
import transforms3d

from hand_teleop.env.sim_env.base import BaseSimulationEnv

from hand_teleop.real_world import lab

from hand_teleop.utils.render_scene_utils import set_entity_color
from hand_teleop.utils.ycb_object_utils import load_ycb_object, YCB_SIZE, YCB_ORIENTATION, YCB_OBJECT_NAMES_EXIT_LIST
from hand_teleop.utils.egad_object_utils import load_egad_object, EGAD_NAME, EGAD_ORIENTATION, EGAD_KIND, EGAD_LIST
from hand_teleop.utils.shapenet_utils import load_shapenet_object, SHAPENET_CAT, CAT_DICT
from hand_teleop.utils.modelnet_object_utils import load_modelnet_object, MODELNET40_ANYTRAIN
from hand_teleop.utils.partnet_utils import load_partnet_object, PARTNET_ORIENTATION, load_car_object
from hand_teleop.utils.assembling_utils import load_assembling_kits_object

from scipy.spatial.transform import Rotation as R


class RelocateEnv(BaseSimulationEnv):

    def __init__(self,
                 use_gui=True,
                 frame_skip=5,
                 object_category="YCB",
                 object_name="tomato_soup_can",
                 object_scale=1.0,
                 randomness_scale=1,
                 friction=1,
                 use_visual_obs=False,
                 **renderer_kwargs):
        super().__init__(use_gui=use_gui,
                         frame_skip=frame_skip,
                         use_visual_obs=use_visual_obs,
                         **renderer_kwargs)

        # Object info
        self.object_category = object_category
        self.object_name = object_name
        self.object_scale = object_scale
        self.object_height = object_scale * YCB_SIZE[self.object_name][2] / 2
        self.target_pose = sapien.Pose()

        # Dynamics info
        self.randomness_scale = randomness_scale
        self.friction = friction

        # Construct scene
        self.scene = self.engine.create_scene()
        self.scene.set_timestep(0.004)

        # Dummy camera creation to initial geometry object
        if self.renderer and not self.no_rgb:
            cam = self.scene.add_camera("init_not_used",
                                        width=10,
                                        height=10,
                                        fovy=1,
                                        near=0.1,
                                        far=1)
            self.scene.remove_camera(cam)

        # Load table
        self.tables = self.create_table(table_height=0.6,
                                        table_half_size=[0.65, 0.65, 0.025])

        # Load object
        if self.object_category.lower() == "ycb":
            self.manipulated_object = load_ycb_object(self.scene, object_name)
            self.target_object = load_ycb_object(self.scene,
                                                 object_name,
                                                 visual_only=True)
            if self.use_visual_obs:
                self.target_object.hide_visual()
        else:
            raise NotImplementedError
        if self.use_gui:
            set_entity_color([self.target_object], [0, 1, 0, 0.6])

    def generate_random_object_pose(self, randomness_scale):
        pos = self.np_random.uniform(low=-0.1, high=0.1,
                                     size=2) * randomness_scale
        ycb_orientation = YCB_ORIENTATION[self.object_name]
        position = np.array([pos[0], pos[1], self.object_height])
        pose = sapien.Pose(position, ycb_orientation)
        return pose

    def generate_random_target_pose(self, randomness_scale):
        pos = self.np_random.uniform(low=-0.2, high=0.2,
                                     size=2) * randomness_scale
        height = 0.35
        position = np.array([pos[0], pos[1], height])
        euler = self.np_random.uniform(low=np.deg2rad(-15),
                                       high=np.deg2rad(15),
                                       size=2)
        ycb_orientation = YCB_ORIENTATION[self.object_name]
        quaternion = transforms3d.euler.euler2quat(euler[0], euler[1], 0)
        pose = sapien.Pose(
            position,
            transforms3d.quaternions.qmult(ycb_orientation, quaternion))
        return pose

    def reset_env(self):
        pose = self.generate_random_object_pose(self.randomness_scale)
        self.manipulated_object.set_pose(pose)

        # Target pose
        pose = self.generate_random_target_pose(self.randomness_scale)
        self.target_object.set_pose(pose)
        self.target_pose = pose


class LabRelocateEnv(BaseSimulationEnv):

    def __init__(self,
                 use_gui=True,
                 frame_skip=10,
                 object_category="YCB",
                 object_name="tomato_soup_can",
                 randomness_scale=1,
                 friction=1,
                 use_visual_obs=False,
                 use_orientation=False,
                 **renderer_kwargs):

        super().__init__(use_gui=use_gui,
                         frame_skip=frame_skip,
                         use_visual_obs=use_visual_obs,
                         **renderer_kwargs)

        # Object info
        self.object_category = object_category
        self.object_name = object_name
        self.object_scale = 1
        self.target_pose = sapien.Pose()

        # Dynamics info
        self.randomness_scale = randomness_scale
        self.friction = friction
        self.use_orientation = use_orientation

        # Construct scene
        self.scene = self.engine.create_scene()
        self.scene.set_timestep(0.005)

        # Dummy camera creation to initial geometry object
        if self.renderer and not self.no_rgb:
            cam = self.scene.add_camera("init_not_used",
                                        width=10,
                                        height=10,
                                        fovy=1,
                                        near=0.1,
                                        far=1)
            self.scene.remove_camera(cam)

        # # Load table
        self.tables = self.create_lab_tables(table_height=0.6)

        # Load object
        self.manipulated_object, self.target_object, self.object_height = self.load_object(
            object_name)
        #===========================tactile============================
        self.obj_index = 0
        self.init_obj_pose = [0, 0, 0]
        #===========================tactile============================

    def load_object(self, object_name):
        if self.object_category.lower() == "ycb":
            #===========================any============================
            # manipulated_object = load_ycb_object(self.scene, object_name)
            # target_object = load_ycb_object(self.scene,
            #                                 object_name,
            #                                 visual_only=True)
            # target_object.set_name("target_object")
            # object_height = YCB_SIZE[self.object_name][2] / 2

            if self.object_name == "any_eval" or self.object_name == "any_train":
                names = YCB_OBJECT_NAMES_EXIT_LIST
                object_name = self.np_random.choice(names)
                #===========================tactile============================
                self.obj_index = names.index(object_name)
                self.temp_object_name = object_name
                #===========================tactile============================

            manipulated_object = load_ycb_object(self.scene, object_name)
            target_object = load_ycb_object(self.scene,
                                            object_name,
                                            visual_only=True)
            target_object.set_name("target_object")
            object_height = YCB_SIZE[object_name][2] / 2

            #===========================any============================
        elif self.object_category.lower() == "egad":

            if self.object_name == "any_train_down":
                names = EGAD_LIST["down"]
                object_name = self.np_random.choice(names)

            if self.object_name == "any_train_front":
                names = EGAD_LIST["front"]
                object_name = self.np_random.choice(names)

            elif len(self.object_name) == 1:
                names = EGAD_KIND[self.object_name]
                object_name = self.np_random.choice(names)
                #===========================tactile============================
                self.obj_index = names.index(object_name)
                #===========================tactile============================

            manipulated_object, self.scale = load_egad_object(
                self.scene, model_id=object_name)
            target_object, self.scale = load_egad_object(self.scene,
                                                         model_id=object_name,
                                                         visual_only=True)
            target_object.set_name("target_object")
            object_height = 0.10
            self.temp_object_name = object_name

        elif self.object_category.isnumeric():
            if self.object_category not in SHAPENET_CAT:
                raise ValueError(
                    f"Object category not recognized: {self.object_category}")
            if self.object_name == "any_eval":
                names = CAT_DICT[self.object_category]["eval"]
                object_name = self.np_random.choice(names)
            if self.object_name == "any_train":
                names = CAT_DICT[self.object_category]["train"]
                object_name = self.np_random.choice(names)

                #===========================tactile============================
                self.obj_index = names.index(object_name)
                #===========================tactile============================

            manipulated_object, object_height = load_shapenet_object(
                self.scene, cat_id=self.object_category, model_id=object_name)
            target_object, _ = load_shapenet_object(
                self.scene,
                cat_id=self.object_category,
                model_id=object_name,
                visual_only=True)
            target_object.set_name("target_object")

        elif self.object_category.lower() == "assembling_kits":
            manipulated_object, object_height = load_assembling_kits_object(
                self.scene, "assembling_kits", self.object_name)
            set_entity_color([manipulated_object], [1, 0, 0, 1])

            target_object, object_height = load_assembling_kits_object(
                self.scene,
                "assembling_kits",
                self.object_name,
                visual_only=True)
            set_entity_color([target_object], [0, 0, 1, 0.7])

        elif self.object_category.lower() == "modelnet":

            if self.object_name == "any_train":
                names = MODELNET40_ANYTRAIN
                object_name = self.np_random.choice(names)
                self.obj_index = names.index(object_name)
            manipulated_object, object_height = load_modelnet_object(
                self.scene, model_name=object_name)

            target_object, object_height = load_modelnet_object(
                self.scene, model_name=object_name, visual_only=True)
            target_object.set_name("target_object")
        elif self.object_category == "partnet":
            manipulated_object, object_height = load_partnet_object(
                self.scene, model_name=object_name)

            target_object = None
        elif self.object_category == "car":
            manipulated_object, object_height = load_car_object(
                self.scene, model_name=object_name)

            target_object = None

        else:
            raise NotImplementedError

        if self.use_visual_obs:
            target_object.hide_visual()
        if self.renderer and not self.no_rgb:
            if target_object is not None:
                set_entity_color([target_object], [0, 1, 0, 0.6])

        return manipulated_object, target_object, object_height

    def generate_random_object_pose(self, randomness_scale):
        pos = self.np_random.uniform(low=-0.1, high=0.1,
                                     size=2) * randomness_scale
        orientation = [1, 0, 0, 0]
        if self.object_category == "ycb":
            #================any====================
            if self.object_name == "any_train" or self.object_name == "any_eval":
                orientation = YCB_ORIENTATION[self.temp_object_name]
            #================any====================
            else:
                orientation = YCB_ORIENTATION[self.object_name]
        # #============== add orientation===================
        elif self.object_category == "egad" and self.temp_object_name in EGAD_ORIENTATION:

            orientation = EGAD_ORIENTATION[self.temp_object_name]
        # #============== add orientation===================
        elif self.object_category == "partnet":
            orientation = PARTNET_ORIENTATION[self.object_name]

        else:
            if self.use_orientation:

                from scipy.spatial.transform import Rotation as R
                z_angles = self.np_random.uniform(low=0, high=1, size=36)
                z_angle = np.random.choice(z_angles)
                r = R.from_rotvec(2 * np.pi * np.array([0, 0, z_angle]))
                self.reorientation = r.as_quat()[[3, 0, 1,
                                                  2]]  #scipy:xyzw, sapien:wxyz
                orientation = self.reorientation

        position = np.array([pos[0], pos[1], self.object_height])

        if self.object_category == "assembling_kits":
            from scipy.spatial.transform import Rotation as R
            r = R.from_euler('xyz', [0, -90, 180], degrees=True)
            orientation = r.as_quat()[[3, 0, 1, 2]]  #x,y,z,w

            # position = np.array([x, y, self.object_height])
            position = np.array([0, 0, self.object_height])

        pose = sapien.Pose(position, orientation)

        return pose

    def generate_random_target_pose(self, randomness_scale):
        pos = self.np_random.uniform(low=-0.2, high=0.2,
                                     size=2) * randomness_scale
        height = 0.25
        position = np.array([pos[0], pos[1], height])
        orientation = [1, 0, 0, 0]
        # No randomness for the orientation. Keep the canonical orientation.
        if self.object_category == "ycb":
            #================any====================
            if self.object_name == "any_train" or self.object_name == "any_eval":
                orientation = YCB_ORIENTATION[self.temp_object_name]
            #================any====================
            else:
                orientation = YCB_ORIENTATION[self.object_name]
        # #============== add orientation===================
        elif self.object_category == "egad" and self.temp_object_name in EGAD_ORIENTATION:
            orientation = EGAD_ORIENTATION[self.temp_object_name]
        # #============== add orientation===================
        elif self.object_category == "partnet":
            orientation = PARTNET_ORIENTATION[self.object_name]
            # position = self.init_obj_pose
        else:
            if self.use_orientation:
                orientation = self.reorientation

        if self.object_category == "assembling_kits":
            from scipy.spatial.transform import Rotation as R
            r = R.from_euler('xyz', [0, -90, 180], degrees=True)
            orientation = r.as_quat()[[3, 0, 1, 2]]  #x,y,z,w

            # position = np.array([x, y, self.object_height])
            position = np.array([0, 0, 0.2])

        pose = sapien.Pose(position, orientation)

        return pose

    def reset_env(self):
        if "any" in self.object_name or len(self.object_name) == 1:
            self.scene.remove_actor(self.manipulated_object)
            self.scene.remove_actor(self.target_object)
            self.manipulated_object, self.target_object, self.object_height = self.load_object(
                self.object_name)

        pose = self.generate_random_object_pose(self.randomness_scale)
        self.init_obj_pose = pose.p

        self.manipulated_object.set_pose(pose)

        pose = sapien.Pose([pose.p[0], pose.p[1], 0.2])
        if self.target_object is not None:

            self.target_object.set_pose(pose)
        self.target_pose = pose

        if self.object_category == "egad" or self.object_category == "assembling_kits":
            for _ in range(100):

                self.robot.set_qf(
                    self.robot.compute_passive_force(
                        external=False, coriolis_and_centrifugal=False))
                self.scene.step()
            self.object_height = self.manipulated_object.get_pose().p[2]
            self.init_obj_pose = self.manipulated_object.get_pose().p

        # check_contact_links = self.finger_contact_links
        # self.check_actor_pair_contacts(check_contact_links,
        #                                self.manipulated_object)

    def create_lab_tables(self, table_height):
        # Build object table first
        builder = self.scene.create_actor_builder()
        table_thickness = 0.03

        # Top
        top_pose = sapien.Pose(
            np.array([
                lab.TABLE_ORIGIN[0], lab.TABLE_ORIGIN[1], -table_thickness / 2
            ]))
        top_material = self.scene.create_physical_material(1, 0.5, 0.01)

        table_half_size = np.concatenate(
            [lab.TABLE_XY_SIZE / 2, [table_thickness / 2]])

        builder.add_box_collision(pose=top_pose,
                                  half_size=table_half_size,
                                  material=top_material)
        # Leg
        if self.renderer and not self.no_rgb:
            table_visual_material = self.renderer.create_material()
            table_visual_material.set_metallic(0.0)
            table_visual_material.set_specular(0.3)
            table_visual_material.set_base_color(np.array([0.9, 0.9, 0.9, 1]))
            table_visual_material.set_roughness(0.3)

            leg_size = np.array(
                [0.025, 0.025, (table_height / 2 - table_half_size[2])])
            leg_height = -table_height / 2 - table_half_size[2]
            x = table_half_size[0] - 0.1
            y = table_half_size[1] - 0.1

            builder.add_box_visual(pose=top_pose,
                                   half_size=table_half_size,
                                   material=table_visual_material)
            builder.add_box_visual(pose=sapien.Pose(
                [x, y + lab.TABLE_ORIGIN[1], leg_height]),
                                   half_size=leg_size,
                                   material=table_visual_material,
                                   name="leg0")
            builder.add_box_visual(pose=sapien.Pose(
                [x, -y + lab.TABLE_ORIGIN[1], leg_height]),
                                   half_size=leg_size,
                                   material=table_visual_material,
                                   name="leg1")
            builder.add_box_visual(pose=sapien.Pose(
                [-x, y + lab.TABLE_ORIGIN[1], leg_height]),
                                   half_size=leg_size,
                                   material=table_visual_material,
                                   name="leg2")
            builder.add_box_visual(pose=sapien.Pose(
                [-x, -y + lab.TABLE_ORIGIN[1], leg_height]),
                                   half_size=leg_size,
                                   material=table_visual_material,
                                   name="leg3")
        object_table = builder.build_static("object_table")

        # Build robot table
        table_half_size = np.array([0.3, 0.8, table_thickness / 2])
        robot_table_offset = -lab.DESK2ROBOT_Z_AXIS - 0.004
        table_height += robot_table_offset
        builder = self.scene.create_actor_builder()
        top_pose = sapien.Pose(
            np.array([
                lab.ROBOT2BASE.p[0] - table_half_size[0] + 0.08,
                lab.ROBOT2BASE.p[1] - table_half_size[1] + 0.08,
                -table_thickness / 2 + robot_table_offset
            ]))
        top_material = self.scene.create_physical_material(1, 0.5, 0.01)
        builder.add_box_collision(pose=top_pose,
                                  half_size=table_half_size,
                                  material=top_material)
        if self.renderer and not self.no_rgb:
            table_visual_material = self.renderer.create_material()
            table_visual_material.set_metallic(0.0)
            table_visual_material.set_specular(0.5)
            table_visual_material.set_base_color(
                np.array([239, 212, 151, 255]) / 255)
            table_visual_material.set_roughness(0.1)
            builder.add_box_visual(pose=top_pose,
                                   half_size=table_half_size,
                                   material=table_visual_material)
        robot_table = builder.build_static("robot_table")
        return object_table, robot_table


def env_test():
    from sapien.utils import Viewer

    env = LabRelocateEnv(object_category="partnet",
                         object_name="100335",
                         use_orientation=True)
    # env = LabRelocateEnv(object_category="02876657", object_name="any_train")
    viewer = Viewer(env.renderer)
    viewer.set_scene(env.scene)
    viewer.set_camera_rpy(r=0, p=0, y=-np.pi)
    add_default_scene_light(env.scene, env.renderer)
    env.viewer = viewer

    env.reset_env()
    frame = 0
    while not viewer.closed:
        frame += 1
        # if frame % 100 == 0:
        #     env.reset_env()

        #env.simple_step()
        env.render()


if __name__ == '__main__':
    env_test()
