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
from hand_teleop.utils.partnet_utils import PARTNET_ORIENTATION, load_car_object, load_partnet_actor, load_spoon_object, SPOON_ANYTRAIN, PEN_ANYTRAIN_NOVEL, PEN_ANYTRAIN2, load_partnet_arti2actor, USB_ANYTRAIN
from hand_teleop.utils.assembling_utils import load_assembling_kits_object
from hand_teleop.utils.tactile_loader_utils import save_spoon_mesh, save_partnet_actor_mesh
from scipy.spatial.transform import Rotation as R

from copy import deepcopy

import os


class LabInsertionEnv(BaseSimulationEnv):

    def __init__(self,
                 use_gui=True,
                 frame_skip=10,
                 object_category="YCB",
                 object_name="any_train",
                 randomness_scale=1,
                 friction=1,
                 use_visual_obs=False,
                 use_orientation=False,
                 render_mesh=False,
                 novel=False,
                 **renderer_kwargs):

        super().__init__(use_gui=use_gui,
                         frame_skip=frame_skip,
                         use_visual_obs=use_visual_obs,
                         **renderer_kwargs)

        # Object info
        self.object_category = object_category
        self.novel = novel

        self.object_name = object_name
        self.object_scale = 1
        self.target_pose = sapien.Pose()

        # Dynamics info
        self.randomness_scale = randomness_scale
        self.friction = friction
        self.use_orientation = use_orientation
        self.render_mesh = render_mesh

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

        self.pen_list = [
            '101736', '101714', '101713', '101712', '101703', '101698',
            '101722'
        ]

        self.box = None
        # Load object
        self.manipulated_object, self.target_object, self.object_height = self.load_object(
            object_name)

        if self.render_mesh:
            save_partnet_mesh(self.manipulated_object, self.object_category,
                              str(object_name))
        #===========================tactile============================
        self.obj_index = 0
        self.init_obj_pose = [0, 0, 0]
        #===========================tactile============================
        # self.create_bg()

    def create_hole(self, object_height):
        builder = self.scene.create_actor_builder()

        box_height = self.np_random.uniform(low=object_height / 3,
                                            high=object_height * 0.7,
                                            size=1)[0]

        box_with = 0.03
        box_hole_x = self.object_length[1] + 0.006
        box_hole_y = self.object_length[0] + 0.006


        # left
        half_size = [box_height, (box_with - box_hole_y) / 4,
                     box_hole_x / 2]  #[z,y,x]
        pose = sapien.Pose(
            p=[-box_height, -(box_with - box_hole_y) / 4 - box_hole_x / 2, 0],
            q=[1, 0, 0, 0])
        builder.add_box_collision(half_size=half_size, pose=pose)
        builder.add_box_visual(half_size=half_size, pose=pose)

        # # right
        half_size = [box_height, (box_with - box_hole_y) / 4,
                     box_hole_x / 2]  #[z,y,x]
        pose = sapien.Pose(
            p=[-box_height, (box_with - box_hole_y) / 4 + box_hole_x / 2, 0],
            q=[1, 0, 0, 0])
        builder.add_box_collision(half_size=half_size, pose=pose)
        builder.add_box_visual(half_size=half_size, pose=pose)

        # # # front
        half_size = [box_height, (box_with) / 2,
                     (box_with - box_hole_x) / 4]  #[z,y,x]
        pose = sapien.Pose(
            p=[-box_height, 0, (box_with - box_hole_x) / 4 + box_hole_y / 2],
            q=[1, 0, 0, 0])
        builder.add_box_collision(half_size=half_size, pose=pose)
        builder.add_box_visual(half_size=half_size, pose=pose)

        # # back
        half_size = [box_height, (box_with) / 2,
                     (box_with - box_hole_x) / 4]  #[z,y,x]
        pose = sapien.Pose(
            p=[-box_height, 0, -(box_with - box_hole_x) / 4 - box_hole_y / 2],
            q=[1, 0, 0, 0])
        builder.add_box_collision(half_size=half_size, pose=pose)
        builder.add_box_visual(half_size=half_size, pose=pose)

        # # bottom
        half_size = [0.001, box_hole_y, box_hole_x]  #[z,y,x]
        pose = sapien.Pose(p=[0, 0, 0], q=[1, 0, 0, 0])
        builder.add_box_collision(half_size=half_size, pose=pose)
        builder.add_box_visual(half_size=half_size, pose=pose)

        self.hole_height = self.np_random.uniform(low=0, high=0.2,
                                                  size=1)[0] * 1

        # add hole height
        half_size = [
            self.hole_height / 2, box_with / 2 + 0.02, box_with / 2 + 0.02
        ]  #[z,y,x]
        pose = sapien.Pose(p=[self.hole_height / 2, 0, 0], q=[1, 0, 0, 0])
        builder.add_box_collision(half_size=half_size, pose=pose)
        builder.add_box_visual(half_size=half_size, pose=pose)

        self.box = builder.build_static()
        r = R.from_euler('xyz', [0, 90, 0], degrees=True)
        orientation = r.as_quat()[[3, 0, 1, 2]]  #x,y,z,w
        self.box.set_pose(
            sapien.Pose(p=[0, 0, self.hole_height], q=orientation))
        set_entity_color([self.box], [0.8, 0.9, 0.8, 1])

        return self.box, box_height

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

    def load_object(self, obj_name):

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
            self.object_length = 2 * self.object_length

            self.box, box_height = self.create_hole(object_height)

            self.floor_height = box_height
            self.center = [0, 0, self.floor_height]

        elif self.object_category in ["USB"]:

            names = USB_ANYTRAIN
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
                height=height,
                insertion=True)

            target_object, self.scale, object_height, _ = load_partnet_arti2actor(
                self.scene,
                self.object_category,
                model_name=object_name,
                visual_only=True,
                height=height,
                insertion=True)

            self.box, box_height = self.create_hole(object_height)

            self.floor_height = box_height
            self.center = [0, 0, self.floor_height]

        elif self.object_category in ["pen"]:

            names = PEN_ANYTRAIN2

            if self.novel:
                names = PEN_ANYTRAIN_NOVEL
            height = self.np_random.uniform(low=0.13, high=0.18, size=1)[0]

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

            target_object, self.scale, object_height, self.object_length = load_partnet_actor(
                self.scene,
                self.object_category,
                model_name=object_name,
                visual_only=True,
                height=height)

            self.box, box_height = self.create_hole(object_height)

            # self.box.set_pose(sapien.Pose([0, 0, 0], [0.707, 0, 0.707, 0]))

            self.floor_height = box_height
            self.center = [0, 0, self.floor_height]
            if self.render_mesh:
                save_partnet_actor_mesh(manipulated_object, object_name)
        target_object.hide_visual()

        return manipulated_object, target_object, object_height

    def create_pan(self, object_length, visual_only=False):

        builder = self.scene.create_actor_builder()

        material = self.scene.create_physical_material(4, 4, 0.01)

        if not visual_only:
            builder.add_multiple_collisions_from_file(
                filename="box.stl",
                scale=[
                    object_length[0] * 2, object_length[1] * 2,
                    object_length[2] * 2
                ],
                pose=sapien.Pose([0, 0, 0]),
                density=1000,
                material=material,
            )

        builder.add_visual_from_file(
            filename="box.stl",
            scale=[
                object_length[0] * 2, object_length[1] * 2,
                object_length[2] * 2
            ],
            pose=sapien.Pose([0, 0, 0]),
        )

        if not visual_only:
            rectangle = builder.build("pan")
        else:
            rectangle = builder.build_static()
        return rectangle

    def build_box(self, inner_radius, height):

        builder = self.scene.create_actor_builder()

        builder.add_box_collision(
            half_size=[inner_radius, inner_radius, height])
        builder.add_box_visual(half_size=[inner_radius, inner_radius, height])
        return builder.build_static("target_object"), height

    def load_partnet_object(self,
                            scene: sapien.Scene,
                            model_name: str,
                            physical_material: sapien.PhysicalMaterial = None,
                            density=1000,
                            visual_only=False):

        filename = "./assets/partnet-mobility-dataset/pen/" + model_name + "/mobility.urdf"
        loader = scene.create_urdf_loader()
        loader.scale = 0.05

        builder = loader.load_file_as_articulation_builder(filename)
        loader.load_multiple_collisions_from_file = True

        if not visual_only:

            object = builder.build(fix_root_link=False)

        return object

    def _build_box_with_hole(self,
                             inner_radius,
                             outer_radius,
                             depth,
                             center=(0, 0),
                             name="box_with_hole",
                             height=0.03125 / 1.1,
                             static=False):
        if static:
            builder = self.scene.create_actor_builder()
        else:
            box_builder: sapien.ArticulationBuilder = self.scene.create_articulation_builder(
            )
            builder = box_builder.create_link_builder()

        thickness = (outer_radius - inner_radius) * 0.5
        # x-axis is hole direction
        half_center = [x * 0.5 for x in center]
        self.center = center
        half_sizes = [
            [depth, thickness - half_center[0], outer_radius],
            [depth, thickness + half_center[0], outer_radius],
            [depth, outer_radius, thickness - half_center[1]],
            [depth, outer_radius, thickness + half_center[1]],
        ]
        offset = thickness + inner_radius
        poses = [
            sapien.Pose([0, offset + half_center[0], 0]),
            sapien.Pose([0, -offset + half_center[0], 0]),
            sapien.Pose([0, 0, offset + half_center[1]]),
            sapien.Pose([0, 0, -offset + half_center[1]]),
        ]

        # mat = self.renderer.create_material()
        # # mat.set_base_color(hex2rgba("#FFD289"))
        # mat.metallic = 0.0
        # mat.roughness = 0.5
        # mat.specular = 0.5

        half_sizes = np.array(half_sizes)

        half_sizes[:, 0] = height
        self.floor_height = height

        for (half_size, pose) in zip(half_sizes, poses):

            builder.add_box_collision(pose, half_size)
            builder.add_box_visual(pose, half_size)
        if static:
            return builder.build_static(name), half_sizes[0, 0] / 1.25
        else:

            builder.set_name("box")
            handdle_builder = box_builder.create_link_builder()

            half_size[0] = half_size[0] * 1
            half_size[1] = half_size[1] * 1
            half_size[2] = half_size[2] * 2
            pose = sapien.Pose([0, 0, -offset - half_sizes[-1, 1]],
                               [0.707, 0.707, 0, 0])
            self.handle_offset = -offset - half_sizes[-1, 1]

            handdle_builder.add_box_collision(pose, half_size)
            handdle_builder.add_box_visual(pose, half_size)
            handdle_builder.set_name("handle")

            handdle_builder.set_joint_properties(joint_type="fixed",
                                                 limits=[],
                                                 pose_in_parent=sapien.Pose(
                                                     [0, 0, 0], [1, 0, 0, 0]))
            builder.set_parent(handdle_builder.get_index())

            for link in box_builder.get_link_builders():
                link.set_collision_groups(1, 1, 4, 4)

            return box_builder.build(fix_root_link=True), half_sizes[0, 0]

    def generate_random_object_pose(self, randomness_scale):
        pos = np.zeros(2)
        pos[1] = self.np_random.uniform(
            low=0.03, high=0.06,
            size=1) * randomness_scale * self.np_random.choice([-1, 1])
        pos[0] = self.np_random.uniform(low=-0.05, high=0.0,
                                        size=1) * randomness_scale
        # pos = self.np_random.uniform(low=-0.1, high=0.1,
        #                              size=2) * randomness_scale
        # pos = np.array([self.center[0], self.center[1], self.object_height])

        orientation = [1, 0, 0, 0]

        position = np.array([
            pos[0] * 1, pos[1] * 1, self.object_height +
            2 * self.floor_height + self.hole_height + 0.05
        ])  #+ self.floor_height * 2
        # if self.object_category in ["USB"]:
        # r = R.from_euler('xyz', [0, 0, 90], degrees=True)

        # orientation = r.as_quat()[[3, 0, 1, 2]]  #x,y,z,w

        pose = sapien.Pose(position, orientation)

        return pose

    def generate_random_target_pose(self, randomness_scale):
        pos = self.np_random.uniform(low=-0.1, high=0.1,
                                     size=2) * randomness_scale
        # height = 0.35
        position = np.array([
            self.center[0], self.center[1],
            self.object_height + self.hole_height
        ])

        from scipy.spatial.transform import Rotation as R
        self.rotation_factor = self.np_random.uniform(low=-1, high=1,
                                                      size=1) * 0

        r = R.from_euler('xyz', [0, 0, self.rotation_factor * 90],
                         degrees=True)

        # if self.object_category in ["USB"]:
        #     r = R.from_euler('xyz', [0, 0, 90], degrees=True)

        orientation = r.as_quat()[[3, 0, 1, 2]]  #x,y,z,w

        pose = sapien.Pose(position, orientation)

        return pose

    def reset_env(self):
        if "any" in self.object_name or len(self.object_name) == 1:
            # self.scene.remove_articulation(self.manipulated_object)
            # # self.scene.remove_articulation(self.target_object)
            # self.manipulated_object, self.target_object, self.object_height = self.load_object(
            #     self.object_name)
            self.scene.remove_actor(self.manipulated_object)
            if self.box is not None:
                self.scene.remove_actor(self.box)
            # self.scene.remove_actor(self.holder)

            if self.target_object is not None:
                self.scene.remove_actor(self.target_object)
            self.manipulated_object, self.target_object, self.object_height = self.load_object(
                self.object_name)

        # print(self.target_pose, self.init_obj_pose, self.object_height)

        pose = self.generate_random_object_pose(self.randomness_scale)
        self.init_obj_pose = pose.p

        self.manipulated_object.set_pose(pose)

        self.target_pose = self.generate_random_target_pose(
            self.randomness_scale)

        if self.target_object is not None:

            self.target_object.set_pose(self.target_pose)

            set_entity_color([self.target_object], [0, 1, 0, 0.6])

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
        # set_entity_color([object_table], [0,0,0, 1])

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
