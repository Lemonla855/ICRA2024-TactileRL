import pdb
import numpy as np
import sapien.core as sapien
import transforms3d

from hand_teleop.env.sim_env.base import BaseSimulationEnv

from hand_teleop.real_world import lab

from hand_teleop.utils.render_scene_utils import set_entity_color
from hand_teleop.utils.ycb_object_utils import load_ycb_articulated_object, YCB_SIZE, YCB_ORIENTATION, YCB_OBJECT_NAMES_EXIT_LIST
from hand_teleop.utils.egad_object_utils import load_egad_object, EGAD_NAME, EGAD_ORIENTATION, EGAD_KIND
from hand_teleop.utils.shapenet_utils import load_shapenet_object, SHAPENET_CAT, CAT_DICT, load_bolt_object, BOLT_NAMES
from hand_teleop.utils.modelnet_object_utils import load_modelnet_object, MODELNET40_ANYTRAIN

from hand_teleop.utils.partnet_utils import load_partnet_articulation, PARTNET_ORIENTATION, PEN_ANYTRAIN
from hand_teleop.utils.assembling_utils import load_assembling_kits_object
from hand_teleop.utils.partnet_utils import load_partnet_object, PARTNET_ORIENTATION, load_spoon_object, SPOON_ANYTRAIN, load_kitchen_object, KITCHEN_ANYTRAIN
from hand_teleop.utils.tactile_loader_utils import save_spoon_mesh


class LabWriteEnv(BaseSimulationEnv):

    def __init__(self,
                 use_gui=True,
                 frame_skip=10,
                 object_category="YCB",
                 object_name="tomato_soup_can",
                 randomness_scale=1,
                 friction=1,
                 use_visual_obs=False,
                 use_orientation=False,
                 render_mesh=False,
                 index=0,
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
        self.render_mesh = render_mesh
        self.index = index

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
        self.box = None
        self.manipulated_object, self.target_object, self.object_height = self.load_object(
            object_name)

        if self.box is not None:
            self.scene.remove_actor(self.box)

        #===========================tactile============================
        self.obj_index = 0
        self.init_obj_pose = [0, 0, 0]

        self.floor_height = 0
        self.rotation_factor = 0
        #===========================tactile============================

    def load_annulus(self, object_height):
        builder = self.scene.create_actor_builder()
        # outer
        builder.add_multiple_collisions_from_file(
            "annulus.stl", scale=[0.1, 0.1, object_height])
        builder.add_visual_from_file("annulus.stl",
                                     scale=[0.1, 0.1, object_height])

        #inner

        builder.add_multiple_collisions_from_file(
            "annulus.stl",
            scale=[
                0.1 - self.object_length[1] * 2.3,
                0.1 - self.object_length[1] * 2.3, object_height
            ])
        builder.add_visual_from_file("annulus.stl",
                                     scale=[
                                         0.1 - self.object_length[1] * 2.3,
                                         0.1 - self.object_length[1] * 2.3,
                                         object_height
                                     ])

        annulus = builder.build_static("annulus")
        return annulus

    def create_pan(self, object_height):

        self.floor_height = object_height / 5

        width = 0.04

        builder = self.scene.create_actor_builder()
        builder.add_box_collision(half_size=[width, width, self.floor_height],
                                  pose=sapien.Pose([0, 0, self.floor_height]))
        builder.add_box_visual(half_size=[width, width, self.floor_height],
                               pose=sapien.Pose([0, 0, self.floor_height]))

        self.box = builder.build_static("pan")

    def load_object(self, object_name):

        if self.object_category.lower() == "spoon":

            if self.object_name == "any_eval" or self.object_name == "any_train":

                names = SPOON_ANYTRAIN
                object_name = self.np_random.choice(names)
                #===========================tactile============================
                self.obj_index = names.index(object_name)
                self.temp_object_name = object_name

            manipulated_object, self.scale, object_height, self.object_length = load_spoon_object(
                self.scene, object_name)
            target_object, _, _, _ = load_spoon_object(self.scene,
                                                       object_name,
                                                       visual_only=True)
            target_object.set_name("target_object")
            # target_object = None
            if self.render_mesh:
                save_spoon_mesh(manipulated_object, object_name)

            self.floor_height = 0
            self.center = [0, 0, self.floor_height]

        if self.object_category.lower() == "kitchen":

            if self.object_name == "any_eval" or self.object_name == "any_train":

                names = KITCHEN_ANYTRAIN
                object_name = self.np_random.choice(names)
                #===========================tactile============================
                self.obj_index = names.index(object_name)
                self.temp_object_name = object_name

            manipulated_object, self.scale, object_height, self.object_length = load_kitchen_object(
                self.scene, object_name)
            target_object, _, _, _ = load_kitchen_object(self.scene,
                                                         object_name,
                                                         visual_only=True)
            target_object.set_name("target_object")
            if self.render_mesh:
                save_spoon_mesh(manipulated_object, object_name)

            self.floor_height = 0
            self.center = [0, 0, self.floor_height]

        if self.object_category in ["assembling_kits"]:
            manipulated_object, _, object_height, self.object_length = load_assembling_kits_object(
                self.scene, model_id=object_name)
            target_object, _, _, _ = load_assembling_kits_object(
                self.scene, model_id=object_name, visual_only=True)

        if self.object_category in ["pen", "knife"]:

            if self.object_name == "any_eval" or self.object_name == "any_train":
                names = PEN_ANYTRAIN
                object_name = self.np_random.choice(names)
                #===========================tactile============================
                self.obj_index = names.index(object_name)
                self.temp_object_name = object_name

            manipulated_object, self.scale, object_height, self.object_length = load_partnet_articulation(
                self.scene,
                self.object_category,
                model_name=object_name,
            )

            target_object, self.scale, object_height, _ = load_partnet_articulation(
                self.scene,
                self.object_category,
                model_name=object_name,
                visual_only=True)

            # self.build_hole(self.object_length, height)
            self.floor_height = 0
            self.center = [0, 0, self.floor_height]
            # self.annulus = self.load_annulus(object_height)
            # self.create_pan(object_height)

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
            else:
                self.temp_object_name = object_name

            manipulated_object = load_ycb_articulated_object(
                self.scene, object_name)
            target_object = load_ycb_articulated_object(self.scene,
                                                        object_name,
                                                        visual_only=True)
            target_object.set_name("target_object")
            target_object = None
            object_height = YCB_SIZE[object_name][2] / 2

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

            manipulated_object, object_height, self.scale, self.object_length = load_shapenet_object(
                self.scene,
                cat_id=self.object_category,
                model_id=object_name,
            )
            target_object, _, _, _ = load_shapenet_object(
                self.scene,
                cat_id=self.object_category,
                model_id=object_name,
                visual_only=True,
            )
            target_object.set_name("target_object")

            # self.create_pan(object_height)

        elif self.object_category.lower() == "bolt":

            if self.object_name == "any_train":
                names = BOLT_NAMES
                object_name = self.np_random.choice(names)

                #===========================tactile============================
                self.obj_index = names.index(object_name)
                #===========================tactile============================

            manipulated_object, object_height, self.scale, self.object_length = load_bolt_object(
                self.scene,
                cat_id=self.object_category,
                model_id=object_name,
            )
            target_object, _, _, _ = load_bolt_object(
                self.scene,
                cat_id=self.object_category,
                model_id=object_name,
                visual_only=True,
            )
            target_object.set_name("target_object")

        elif self.object_category.lower() == "bolt":

            if self.object_name == "any_train":
                names = BOLT_NAMES
                object_name = self.np_random.choice(names)

                #===========================tactile============================
                self.obj_index = names.index(object_name)
                #===========================tactile============================

            manipulated_object, object_height, self.scale, self.object_length = load_bolt_object(
                self.scene,
                cat_id=self.object_category,
                model_id=object_name,
                scale_factor=1.2,
                height_factor=1.0)
            target_object, _, _, _ = load_bolt_object(
                self.scene,
                cat_id=self.object_category,
                model_id=object_name,
                visual_only=True,
                scale_factor=1.2,
                height_factor=1.0)
            target_object.set_name("target_object")

        if self.use_visual_obs:
            target_object.hide_visual()
        if self.renderer and not self.no_rgb:
            if target_object is not None:
                set_entity_color([target_object], [0, 1, 0, 0.6])

        return manipulated_object, target_object, object_height

    def generate_random_object_pose(self, randomness_scale):
        from scipy.spatial.transform import Rotation as R

        pos = np.zeros(2)
        pos[1] = self.np_random.uniform(
            low=0.15, high=0.20,
            size=1) * randomness_scale * self.np_random.choice([-1, 1])
        pos[0] = self.np_random.uniform(low=-0.15, high=0.0,
                                        size=1) * randomness_scale

        if self.object_category in ["spoon"]:
            self.sign_distance = self.np_random.choice([-1, 1])
            # self.sign_distance = -1

            if self.index == 0:
                if self.sign_distance < 0:

                    self.rotation_factor = self.np_random.uniform(low=0.05,
                                                                  high=0.10)
                else:
                    self.rotation_factor = self.np_random.uniform(low=0.05,
                                                                  high=0.60)
                position = np.array([
                    0, 0, (self.object_height) *
                    np.cos(self.sign_distance * self.rotation_factor * 90 /
                           180 * np.pi) + self.object_length[0] / 2
                ])
            elif self.index == 1:
                self.rotation_factor = self.np_random.uniform(low=0.5,
                                                              high=0.90)
                position = np.array([
                    0, 0, (self.object_height) *
                    np.cos(self.sign_distance * self.rotation_factor * 90 /
                           180 * np.pi) + self.object_length[0] / 2
                ])
            elif self.index == 2:
                self.sign_distance = 1
                self.rotation_factor = self.np_random.uniform(low=-0.10,
                                                              high=0.10)
                theta = self.current_step / 200 * np.pi * 2
                x = np.cos(theta)
                y = np.sin(theta)

                position = np.array([
                    x * 0.1, y * 0.1, (self.object_height) *
                    np.cos(self.sign_distance * self.rotation_factor * 90 /
                           180 * np.pi)
                ])

                r = R.from_euler(
                    'xyz',
                    [0, self.sign_distance * self.rotation_factor * 90, 0],
                    degrees=True)
                orientation = r.as_quat()[[3, 0, 1, 2]]

        elif self.object_category in ["kitchen"]:
            self.rotation_factor = -self.np_random.uniform(low=0.03,
                                                           high=0.06) * 1
            r = R.from_euler('xyz', [0, self.rotation_factor * 90, 0],
                             degrees=True)
            orientation = r.as_quat()[[3, 0, 1, 2]]
            position = np.array([0, 0, self.object_height])

        elif self.object_category in ["assembling_kits"]:
            r = R.from_euler('xyz', [90, 90, 0], degrees=True)
            orientation = r.as_quat()[[3, 0, 1, 2]]
            position = np.array([0, 0, self.object_length[0]])

        else:

            self.rotation_factor = -self.np_random.uniform(low=0.05,
                                                           high=0.20) * 1
            r = R.from_euler('xyz', [0, self.rotation_factor * 90, 0],
                             degrees=True)
            orientation = r.as_quat()[[3, 0, 1, 2]]

            # orientation = [1, 0, 0, 0]

            theta = self.current_step / 200 * np.pi * 2
            x = np.cos(theta)
            y = np.sin(theta)
            position = np.array([
                0, 0,
                self.object_height * np.cos(self.rotation_factor * np.pi / 2)
            ])

        pose = sapien.Pose(position, orientation)

        return pose

    def generate_random_target_pose(self, randomness_scale):
        from scipy.spatial.transform import Rotation as R

        x, y = self.np_random.uniform(low=0.05, high=0.15,
                                      size=1), self.np_random.uniform(low=-0.1,
                                                                      high=0.1,
                                                                      size=1)

        position = np.array([0, 0, self.object_length[0]])

        if self.object_category in ["spoon", "kitchen"]:
            rotation_factor = self.np_random.uniform(low=0.5, high=0.9)

            if self.index == 0:
                r = R.from_euler('xyz', [0, -90 * rotation_factor, 0],
                                 degrees=True)
                position = np.array([
                    -self.sign_distance * self.object_height * 0.5, 0,
                    self.object_height * np.cos(
                        (90 * rotation_factor * self.sign_distance) / 180 *
                        np.pi)
                ])
            elif self.index == 1:
                r = R.from_euler('xyz', [0, 0, 0], degrees=True)
                position = np.array([
                    -self.sign_distance * self.object_height * 0.5, 0,
                    self.object_height
                ])

            elif self.index == 2:
                rotation_factor = self.np_random.uniform(low=0.6, high=0.9)
                r = R.from_euler('xyz', [0, -90 * rotation_factor, 0],
                                 degrees=True)
                position = np.array([
                    -self.object_height * 0.5, 0, self.object_height * np.cos(
                        (90 * rotation_factor * self.sign_distance) / 180 *
                        np.pi)
                ])

            orientation = r.as_quat()[[3, 0, 1, 2]]  #x,y,z,w

        elif self.object_category in ["assembling_kits"]:
            r = R.from_euler('xyz', [90, 0, 0], degrees=True)
            orientation = r.as_quat()[[3, 0, 1, 2]]

        else:
            r = R.from_euler('xyz', [0, 0, 0], degrees=True)
            orientation = r.as_quat()[[3, 0, 1, 2]]  #x,y,z,w
            position = np.array([x, y, self.object_length[2]])

        pose = sapien.Pose(position, orientation)

        return pose

    def reset_env(self):
        if "any" in self.object_name:
            if isinstance(self.manipulated_object, sapien.Articulation):
                self.scene.remove_articulation(self.manipulated_object)
            else:
                self.scene.remove_actor(self.manipulated_object)
            self.scene.remove_actor(self.target_object)
            # self.scene.remove_actor(self.annulus)

            if self.box is not None:
                self.scene.remove_actor(self.box)
            self.manipulated_object, self.target_object, self.object_height = self.load_object(
                self.object_name)

        pose = self.generate_random_object_pose(self.randomness_scale)
        self.init_obj_pose = pose.p
        self.init_obj_pos = pose

        self.manipulated_object.set_pose(pose)

        if self.box is not None:
            self.box.set_pose(sapien.Pose([pose.p[0], pose.p[1], 0]))

        # Target pose

        pose = self.generate_random_target_pose(self.randomness_scale)

        if self.target_object is not None:

            self.target_object.set_pose(pose)
        self.target_pose = pose

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
