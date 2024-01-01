import pdb
import numpy as np
import sapien.core as sapien
import transforms3d

from hand_teleop.env.sim_env.base import BaseSimulationEnv

from hand_teleop.real_world import lab

from hand_teleop.utils.render_scene_utils import set_entity_color
from hand_teleop.utils.ycb_object_utils import load_ycb_articulated_object, YCB_SIZE, YCB_ORIENTATION, YCB_OBJECT_NAMES_EXIT_LIST
from hand_teleop.utils.egad_object_utils import load_egad_object, EGAD_NAME, EGAD_ORIENTATION, EGAD_KIND
# from hand_teleop.utils.shapenet_utils import load_shapenet_object, SHAPENET_CAT, CAT_DICT, load_bolt_object, BOLT_NAMES
from hand_teleop.utils.modelnet_object_utils import load_modelnet_object, MODELNET40_ANYTRAIN

from hand_teleop.utils.partnet_utils import load_partnet_actor, load_breaking_bed, BUCKET_ANYTRAIN, BREAKING_BOTTLE_ANYTRAIN, BREAKING_ANYTRAIN, BOTTLE_ANYTRAIN, load_shapenet, MUG_ANYTRAIN, PARTNET_ORIENTATION, PEN_ANYTRAIN2, PEN_ANYTRAIN_NOVEL, SPOON_NOVEL_ANYTRAIN, ANY_KIND, ALL_KIND, USB_ANYTRAIN_NOVEL, USB_ANYTRAIN, load_partnet_arti2actor, load_knife_actor, KNIFE_ANYTRAIN, BOX_ANYTRAIN
from hand_teleop.utils.assembling_utils import load_assembling_kits_object
from hand_teleop.utils.partnet_utils import load_partnet_object, PARTNET_ORIENTATION, load_spoon_object, SPOON_ANYTRAIN, load_kitchen_object, KITCHEN_ANYTRAIN
from hand_teleop.utils.tactile_loader_utils import save_spoon_mesh, save_partnet_actor_mesh


class StablePlaceEnv(BaseSimulationEnv):

    def __init__(self,
                 use_gui=True,
                 frame_skip=10,
                 object_category="pen",
                 object_name="any_train",
                 randomness_scale=1,
                 friction=1,
                 use_visual_obs=False,
                 use_orientation=False,
                 render_mesh=False,
                 novel=False,
                 index=0,
                 **renderer_kwargs):
        super().__init__(use_gui=use_gui,
                         frame_skip=frame_skip,
                         use_visual_obs=use_visual_obs,
                         **renderer_kwargs)

        # Object info
        self.novel = novel
        if "any" in object_category:
            self.all_object_category = "any_kind"
            self.object_category = np.random.choice(ANY_KIND)

        elif "all" in object_category:

            if self.novel:
                self.all_object_category = "all_kind"
                self.object_category = np.random.choice(["spoon", "USB"])
            else:
                self.all_object_category = "all_kind"
                self.object_category = np.random.choice(ALL_KIND)

        else:
            self.all_object_category = None
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

        # self.create_bg()

    def create_pan(self):

        self.floor_height = self.np_random.uniform(low=0.060,
                                                   high=0.080,
                                                   size=1)[0]  #0.08
        width = 0.30
        self.floor_height = 0.001

        builder = self.scene.create_actor_builder()
        builder.add_box_collision(half_size=[width, width, self.floor_height],
                                  pose=sapien.Pose([0, 0, self.floor_height]))
        builder.add_box_visual(half_size=[width, width, self.floor_height],
                               pose=sapien.Pose([0, 0, self.floor_height]))

        self.box = builder.build_static("pan")
        rgbd = np.array([249, 248, 237, 255]) / 255
        set_entity_color([self.box], rgbd)
        self.floor_height = 0.08

    def create_bg(self):

        self.floor_height = 0.6
        width = 0.002

        builder = self.scene.create_actor_builder()
        builder.add_box_collision(half_size=[2, width, self.floor_height],
                                  pose=sapien.Pose([0, 0, self.floor_height]))
        builder.add_box_visual(half_size=[2, width, self.floor_height],
                               pose=sapien.Pose([0, 0, self.floor_height]))
        bg = builder.build_static("bg")
        bg.set_pose(sapien.Pose([0, 0.2, -0.3]))
        set_entity_color([bg], [0, 0, 0, 1])

    def load_object(self, object_name):

        if self.object_category in ["pen"]:

            names = PEN_ANYTRAIN2
            if self.novel:
                names = PEN_ANYTRAIN_NOVEL
            height = self.np_random.uniform(low=0.10, high=0.15, size=1)[0]

            if self.object_name == "any_eval" or self.object_name == "any_train":

                object_name = self.np_random.choice(names)
                #===========================tactile============================
                self.obj_index = names.index(object_name)
                self.temp_object_name = object_name

            manipulated_object, self.scale, object_height, self.object_length = load_partnet_actor(
                self.scene,
                self.object_category,
                model_name=object_name,
                height=height)

            target_object, self.scale, object_height, _ = load_partnet_actor(
                self.scene,
                self.object_category,
                model_name=object_name,
                visual_only=True,
                height=height)

            # self.build_hole(self.object_length, height)

            if self.render_mesh:
                save_partnet_actor_mesh(manipulated_object, object_name)

        elif self.object_category in ["box"]:

            names = BOX_ANYTRAIN

            height = self.np_random.uniform(low=0.10, high=0.15, size=1)[0]

            if self.object_name == "any_eval" or self.object_name == "any_train":

                object_name = self.np_random.choice(names)
                #===========================tactile============================
                self.obj_index = names.index(object_name)
                self.temp_object_name = object_name

            manipulated_object, self.scale, object_height, self.object_length = load_partnet_actor(
                self.scene,
                self.object_category,
                model_name=object_name,
                height=height)

            target_object, self.scale, object_height, _ = load_partnet_actor(
                self.scene,
                self.object_category,
                model_name=object_name,
                visual_only=True,
                height=height)

            # self.build_hole(self.object_length, height)

            if self.render_mesh:
                save_partnet_actor_mesh(manipulated_object, object_name)

            if self.all_object_category is not None:
                if "all" in self.all_object_category:

                    self.obj_index += len(PEN_ANYTRAIN2)

        elif self.object_category in ["bottle", "bucket"]:

            names = BOTTLE_ANYTRAIN

            if self.object_category in ["bucket"]:
                names = BUCKET_ANYTRAIN
            if self.novel:
                names = PEN_ANYTRAIN_NOVEL
            height = self.np_random.uniform(low=0.10, high=0.15, size=1)[0]

            if self.object_name == "any_eval" or self.object_name == "any_train":

                object_name = self.np_random.choice(names)
                #===========================tactile============================
                self.obj_index = names.index(object_name)
                self.temp_object_name = object_name

            manipulated_object, self.scale, object_height, self.object_length = load_partnet_actor(
                self.scene,
                self.object_category,
                model_name=object_name,
                height=height)

            target_object, self.scale, object_height, _ = load_partnet_actor(
                self.scene,
                self.object_category,
                model_name=object_name,
                visual_only=True,
                height=height)

            # self.build_hole(self.object_length, height)

            if self.render_mesh:
                save_partnet_actor_mesh(manipulated_object, object_name)

            if self.all_object_category is not None:
                if "all" in self.all_object_category:
                    if self.novel:
                        self.obj_index += len(PEN_ANYTRAIN_NOVEL) + len(
                            SPOON_ANYTRAIN)

                    else:
                        self.obj_index += len(PEN_ANYTRAIN2) + len(
                            BOX_ANYTRAIN)

        elif self.object_category in ["breaking", "breaking_bottle"]:
            if self.object_category in ["breaking_bottle"]:
                names = BREAKING_BOTTLE_ANYTRAIN

            else:

                names = BREAKING_ANYTRAIN

            height = self.np_random.uniform(low=0.10, high=0.15, size=1)[0]

            if self.object_name == "any_eval" or self.object_name == "any_train":

                object_name = self.np_random.choice(names)
                #===========================tactile============================
                self.obj_index = names.index(object_name)
                self.temp_object_name = object_name

            manipulated_object, self.scale, object_height, self.object_length = load_breaking_bed(
                self.scene,
                self.object_category,
                model_name=object_name,
                height=height)

            target_object, self.scale, object_height, _ = load_breaking_bed(
                self.scene,
                self.object_category,
                model_name=object_name,
                visual_only=True,
                height=height)

            # self.build_hole(self.object_length, height)

            if self.render_mesh:
                save_partnet_actor_mesh(manipulated_object, object_name)

        elif self.object_category in ["USB"]:

            names = USB_ANYTRAIN
            if self.novel:
                names = USB_ANYTRAIN_NOVEL
            height = self.np_random.uniform(low=0.13, high=0.18, size=1)[0]

            if self.object_name == "any_eval" or self.object_name == "any_train":

                object_name = self.np_random.choice(names)
                #===========================tactile============================
                self.obj_index = names.index(object_name)
                self.temp_object_name = object_name

            manipulated_object, self.scale, object_height, self.object_length = load_partnet_arti2actor(
                self.scene,
                self.object_category,
                model_name=object_name,
                height=height)

            target_object, self.scale, object_height, _ = load_partnet_arti2actor(
                self.scene,
                self.object_category,
                model_name=object_name,
                visual_only=True,
                height=height)
            # set_entity_color([manipulated_object], [1, 1, 0, 1.0])

            # self.build_hole(self.object_length, height)

            # if self.render_mesh:
            #     save_partnet_actor_mesh(manipulated_object, object_name)

            # if self.all_object_category is not None:
            #     if "all" in self.all_object_category:
            #         if self.novel:
            #             self.obj_index += len(PEN_ANYTRAIN_NOVEL) + len(
            #                 SPOON_ANYTRAIN)

            #         else:
            #             self.obj_index += len(PEN_ANYTRAIN2) + len(
            #                 SPOON_ANYTRAIN)

        elif self.object_category.isnumeric():

            if self.object_name == "any_train":
                names = MUG_ANYTRAIN
                object_name = self.np_random.choice(names)

                #===========================tactile============================
                self.obj_index = names.index(object_name)
                #===========================tactile============================

            manipulated_object, self.scale, object_height, self.object_length = load_shapenet(
                self.scene,
                model_cat=self.object_category,
                model_name=object_name,
            )
            target_object, _, _, _ = load_shapenet(
                self.scene,
                model_cat=self.object_category,
                model_name=object_name,
                visual_only=True,
            )
            target_object.set_name("target_object")

            # self.create_pan(object_height)

        self.create_pan()

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
            size=1) * randomness_scale * self.np_random.choice([-1])
        pos[0] = self.np_random.uniform(low=-0.15, high=0.0,
                                        size=1) * randomness_scale

        # if self.object_category in ["USB"]:

        #     self.rotation_factor = self.np_random.uniform(
        #         low=0.2, high=0.6) * self.np_random.choice([-1, 1])
        #     r = R.from_euler('xyz', [0, self.rotation_factor * 90, 0],
        #                      degrees=True)
        #     orientation = r.as_quat()[[3, 0, 1, 2]]
        #     position = np.array([
        #         0.05, 0,
        #         self.object_height * np.cos(self.rotation_factor * np.pi / 2) +
        #         2 * self.floor_height + 0.15
        #     ])

        if self.object_category in ["box"]:

            self.rotation_factor = self.np_random.uniform(
                low=0.3, high=0.6) * self.np_random.choice([1])
            r = R.from_euler('xyz', [0, self.rotation_factor * 90, 0],
                             degrees=True)
            orientation = r.as_quat()[[3, 0, 1, 2]]
            position = np.array([
                0.10, 0,
                self.object_height * np.cos(self.rotation_factor * np.pi / 2) +
                2 * self.floor_height +
                self.np_random.uniform(low=0.03, high=0.05, size=1)[0]
            ])

        else:

            self.rotation_factor = self.np_random.uniform(
                low=0.4, high=0.6) * self.np_random.choice([1])

            r = R.from_euler('xyz', [0, self.rotation_factor * 90, 0],
                             degrees=True)
            orientation = r.as_quat()[[3, 0, 1, 2]]
            position = np.array([
                0.10, 0,
                self.object_height * np.cos(self.rotation_factor * np.pi / 2) +
                2 * self.floor_height +
                self.np_random.uniform(low=0.03, high=0.05, size=1)[0] * 0 +
                0.02
            ])

        pose = sapien.Pose(position, orientation)

        return pose

    def generate_random_target_pose(self, randomness_scale):
        from scipy.spatial.transform import Rotation as R

        r = R.from_euler('xyz', [0, 0, 0], degrees=True)
        orientation = r.as_quat()[[3, 0, 1, 2]]  #x,y,z,w
        position = np.array(
            [0.10, 0, self.object_height + 2 * self.floor_height])

        pose = sapien.Pose(position, orientation)

        return pose

    def reset_env(self):
        if "any" in self.object_name:
            if isinstance(self.manipulated_object, sapien.Articulation):
                self.scene.remove_articulation(self.manipulated_object)
            else:
                self.scene.remove_actor(self.manipulated_object)
            self.scene.remove_actor(self.target_object)
            if self.box is not None:
                self.scene.remove_actor(self.box)

            if self.all_object_category is not None:
                if "any" in self.all_object_category:
                    self.object_category = self.np_random.choice(ANY_KIND)
                elif "all" in self.all_object_category:

                    if self.novel:

                        self.object_category = np.random.choice(
                            ["spoon", "USB"])
                    else:

                        self.object_category = np.random.choice(ALL_KIND)

            self.manipulated_object, self.target_object, self.object_height = self.load_object(
                self.object_name)

        pose = self.generate_random_object_pose(self.randomness_scale)
        self.init_obj_pose = pose.p
        self.init_obj_pos = pose

        self.manipulated_object.set_pose(pose)

        # if self.box is not None:
        #     self.box.set_pose(sapien.Pose([pose.p[0], pose.p[1], 0]))

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
            table_visual_material: sapien.RenderMaterial = self.renderer.create_material(
            )
            table_visual_material.set_metallic(0.3)
            table_visual_material.set_specular(0.5)
            table_visual_material.set_base_color(np.array([0.9, 0.9, 0.9, 1]))
            table_visual_material.set_roughness(0.3)
            table_visual_material.set_transmission(0)

            builder.add_box_visual(pose=top_pose,
                                   half_size=table_half_size,
                                   material=table_visual_material)
            # builder.add_box_visual(pose=sapien.Pose(
            #     [x, y + lab.TABLE_ORIGIN[1], leg_height]),
            #                        half_size=leg_size,
            #                        material=table_visual_material,
            #                        name="leg0")
            # builder.add_box_visual(pose=sapien.Pose(
            #     [x, -y + lab.TABLE_ORIGIN[1], leg_height]),
            #                        half_size=leg_size,
            #                        material=table_visual_material,
            #                        name="leg1")
            # builder.add_box_visual(pose=sapien.Pose(
            #     [-x, y + lab.TABLE_ORIGIN[1], leg_height]),
            #                        half_size=leg_size,
            #                        material=table_visual_material,
            #                        name="leg2")
            # builder.add_box_visual(pose=sapien.Pose(
            #     [-x, -y + lab.TABLE_ORIGIN[1], leg_height]),
            #                        half_size=leg_size,
            #                        material=table_visual_material,
            #                        name="leg3")
        object_table = builder.build_static("object_table")
        set_entity_color([object_table], [0, 0, 0, 1])

        # Build robot table
        # table_half_size = np.array([0.3, 0.8, table_thickness / 2])
        # robot_table_offset = -lab.DESK2ROBOT_Z_AXIS - 0.004
        # table_height += robot_table_offset
        # builder = self.scene.create_actor_builder()
        # top_pose = sapien.Pose(
        #     np.array([
        #         lab.ROBOT2BASE.p[0] - table_half_size[0] + 0.08,
        #         lab.ROBOT2BASE.p[1] - table_half_size[1] + 0.08,
        #         -table_thickness / 2 + robot_table_offset
        #     ]))
        # top_material = self.scene.create_physical_material(1, 0.5, 0.01)
        # builder.add_box_collision(pose=top_pose,
        #                           half_size=table_half_size,
        #                           material=top_material)
        # if self.renderer and not self.no_rgb:
        #     table_visual_material = self.renderer.create_material()
        #     table_visual_material.set_metallic(0.0)
        #     table_visual_material.set_specular(0.5)
        #     table_visual_material.set_base_color(
        #         np.array([239, 212, 151, 255]) / 255)
        #     table_visual_material.set_roughness(0.1)
        #     builder.add_box_visual(pose=top_pose,
        #                            half_size=table_half_size,
        #                            material=table_visual_material)
        # robot_table = builder.build_static("robot_table")
        robot_table = None
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
