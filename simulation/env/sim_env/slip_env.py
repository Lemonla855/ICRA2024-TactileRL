import pdb
import numpy as np
import sapien.core as sapien
import transforms3d

from hand_teleop.env.sim_env.base import BaseSimulationEnv

from hand_teleop.real_world import lab

from hand_teleop.utils.render_scene_utils import set_entity_color
from hand_teleop.utils.ycb_object_utils import load_ycb_object, YCB_SIZE, YCB_ORIENTATION, YCB_OBJECT_NAMES_EXIT_LIST
from hand_teleop.utils.egad_object_utils import load_egad_object, EGAD_NAME, EGAD_ORIENTATION, EGAD_KIND
from hand_teleop.utils.shapenet_utils import load_shapenet_object, SHAPENET_CAT, CAT_DICT
from hand_teleop.utils.assembling_utils import load_assembling_kits_object
from hand_teleop.utils.modelnet_object_utils import load_modelnet_object, MODELNET40_ANYTRAIN
from hand_teleop.utils.partnet_class_utils import load_partnet_object
# from hand_teleop.utils.articulated_utils import load_partnet_object, load_assembling_kits_object

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
            set_entity_color([self.target_object], [1, 0, 0, 0.6])

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


class LabSlipRelocateEnv(BaseSimulationEnv):

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

    def create_box(self, visual_only=False):
        builder = self.scene.create_actor_builder()

        object_height = 0.04 * 2
        self.object_x = 0.10 * 2
        self.object_y = 0.03 * 2

        obj_pose = sapien.Pose(np.array([self.object_y, 0, 0]))
        obj_material = self.scene.create_physical_material(1, 0.5, 0.01)

        obj_size = [self.object_x, self.object_y, object_height]

        builder.add_visual_from_file(
            str("box.stl"),
            scale=obj_size,
        )
        if not visual_only:
            builder.add_multiple_collisions_from_file(
                str("box.stl"),
                scale=obj_size,
                material=obj_material,
            )
            collision = builder.build("mainpulated_object")
            set_entity_color([collision], [1, 0, 0, 1])
        else:
            collision = builder.build_static("target_object")
            set_entity_color([collision], [0, 0, 1, 1])

        return collision, object_height / 2

    def load_object(self, object_name):

        if self.object_category == "assembling_kits":

            manipulated_object, object_height = load_assembling_kits_object(
                self.scene, "assembling_kits", self.object_name)
            set_entity_color([manipulated_object], [1, 0, 0, 1])

            target_object, object_height = load_assembling_kits_object(
                self.scene,
                "assembling_kits",
                self.object_name,
                visual_only=True)
            set_entity_color([target_object], [0, 0, 1, 0.7])

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
            manipulated_object.set_name("manipulated_object")
            target_object, _ = load_shapenet_object(
                self.scene,
                cat_id=self.object_category,
                model_id=object_name,
                visual_only=True)
            target_object.set_name("target_object")
            set_entity_color([target_object], [0, 0, 1, 0.5])

        else:  #if self.object_category in ["bottle3", "bottle2"]:

            cap_radius = self.np_random.uniform(low=0.07, high=0.13, size=1)
            cap_heigh = self.np_random.uniform(low=0.05, high=0.07, size=1)
            bottle_height = self.np_random.uniform(low=0.10, high=0.12, size=1)
            scale = [cap_radius, cap_heigh, bottle_height]
            self.scale = scale

            manipulated_object, object_height, _ = load_partnet_object(
                self.scene,
                model_type=self.object_category,
                model_name=object_name,
                scale=scale,
                renderer=self.renderer,
            )
            manipulated_object.set_name("manipulated_object")
            target_object, object_height, self.scale = load_partnet_object(
                self.scene,
                model_type=self.object_category,
                model_name=object_name,
                scale=scale,
                renderer=self.renderer,
                visual_only=True)

            target_object.set_name("target_object")
            object_height = bottle_height
            # target_object = None

        return manipulated_object, target_object, object_height

    def generate_random_object_pose(self, randomness_scale):

        x, y = self.np_random.uniform(low=0.02, high=0.1,
                                      size=1), self.np_random.uniform(low=-0.1,
                                                                      high=0.1,
                                                                      size=1)

        if self.object_category == "assembling_kits":

            r = R.from_euler('xyz', [0, -90, 180], degrees=True)
            orientation = r.as_quat()[[3, 0, 1, 2]]  #x,y,z,w
            position = np.array([0, 0., 0.07])
            # orientation = [1, 0, 0, 0]
        elif self.object_category == "lighter":
            r = R.from_euler('xyz', [0, 0, -90], degrees=True)
            orientation = r.as_quat()[[3, 0, 1, 2]]  #x,y,z,w
            position = np.array([0, 0., 0.07])

        else:
            orientation = [0.707, -0.707, 0, 0]
            # orientation = [1, 0, 0, 0]
            position = np.array([0, 0, 0.15])
            # position = np.array([0, 0., 0.2])

        pose = sapien.Pose(position, orientation)

        return pose

    def generate_random_target_pose(self, randomness_scale):

        position = np.array([0, 0.0, 0.2])

        if self.object_category == "assembling_kits":
            r = R.from_euler('xyz', [90, 0, 90], degrees=True)
            orientation = r.as_quat()[[3, 0, 1, 2]]  #x,y,z,w
            # orientation = [1, 0, 0, 0]
        elif self.object_category == "lighter":
            r = R.from_euler('xyz', [0, 0, -90], degrees=True)
            orientation = r.as_quat()[[3, 0, 1, 2]]  #x,y,z,w
            position = np.array([0, 0., 0.2])
        else:
            orientation = [1, 0, 0, 0]
        pose = sapien.Pose(position, orientation)

        return pose

    def reset_env(self):
        if "any" in self.object_name or len(self.object_name) == 1:
            self.scene.remove_actor(self.manipulated_object)
            self.manipulated_object, self.target_object, self.object_height = self.load_object(
                self.object_name)

        pose = self.generate_random_object_pose(self.randomness_scale)
        self.manipulated_object_pose = pose
        self.init_obj_pose = pose.p

        self.manipulated_object.set_pose(pose)

        # Target pose
        pose = self.generate_random_target_pose(self.randomness_scale)
        if self.target_object is not None:
            self.target_object.set_pose(pose)
        self.target_pose = pose

        if "bottle" in self.object_category:
            self.manipulated_object.get_active_joints()[0].set_limits([[0, 0]])
            self.manipulated_object.get_active_joints()[1].set_limits([[0, 0]])

        if "lighter" in self.object_category:
            self.manipulated_object.get_active_joints()[0].set_limits([[0, 0]])

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
    from constructor import add_default_scene_light
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
