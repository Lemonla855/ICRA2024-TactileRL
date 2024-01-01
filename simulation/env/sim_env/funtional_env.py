import pdb
import numpy as np
import sapien.core as sapien
import transforms3d

from hand_teleop.env.sim_env.base import BaseSimulationEnv

from hand_teleop.real_world import lab

from hand_teleop.utils.render_scene_utils import set_entity_color

from hand_teleop.utils.partnet_class_utils import load_partnet_object, BOTTLE_ANYTRAIN, LIGHTER_ANYTRAIN, DISPENSER_ANYTRAIN, OBJECT_ORIENTATION, KINIFE_ANYTRAIN
from hand_teleop.utils.tactile_loader_parnet_utils import save_partnet_mesh
# from hand_teleop.utils.partnet_utils import load_partnet_object
from scipy.spatial.transform import Rotation as R
import random
from scipy.spatial.transform import Rotation as R


class LabFunctionalEnv(BaseSimulationEnv):
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
        scene_config = sapien.SceneConfig()

        self.scene = self.engine.create_scene()
        self.scene.set_timestep(0.005)
        self.render_mesh = render_mesh

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
        manipulated_object, object_height = load_partnet_object(
            self.scene, self.object_category, model_name=object_name)
        # target_object = load_partnet_object(self.scene,
        #                                     model_name=object_name)
        target_object = None

        if self.use_visual_obs:
            target_object.hide_visual()
        if self.renderer and not self.no_rgb:

            if target_object is not None:
                set_entity_color([target_object], [0, 1, 0, 0.6])
        if self.render_mesh:
            save_partnet_mesh(manipulated_object, self.object_category,
                              object_name)

        return manipulated_object, target_object, object_height

    def generate_random_object_pose(self, randomness_scale):
        pos = self.np_random.uniform(low=-0.1, high=0.1,
                                     size=2) * randomness_scale

        orientation = OBJECT_ORIENTATION[self.object_category]

        # if self.object_name not in ["101052"]:
        #     position = np.array([pos[0], pos[1], self.object_height])
        # else:

        position = np.array([pos[0] * 0, pos[1] * 0, self.object_height])

        # if self.object_name in ["101052"] :
        #     position = [0.136925, 0.00115259, 0.303727]

        pose = sapien.Pose(position, orientation)
        return pose

    def generate_random_target_pose(self, randomness_scale):
        pos = self.np_random.uniform(low=-0.2, high=0.2,
                                     size=2) * randomness_scale
        height = 0.25
        position = np.array([pos[0], pos[1], height])

        if self.use_orientation:
            orientation = self.reorientation
        else:
            orientation = [1, 0, 0, 0]
        pose = sapien.Pose(position, orientation)
        return pose

    def reset_env(self):
        if "any" in self.object_name or len(self.object_name) == 1:
            self.scene.remove_actor(self.manipulated_object)
            self.scene.remove_actor(self.target_object)
            self.manipulated_object, self.target_object, self.object_height = self.load_object(
                self.object_name)

        pose = self.generate_random_object_pose(self.randomness_scale)

        self.manipulated_object.set_pose(pose)
        self.manipulated_object.set_qpos(
            np.zeros(len(self.manipulated_object.get_qpos())))
        # self.init_obj_pose = self.manipulated_object.get_links()[-1].get_pose(
        # ).p
        self.init_obj_pose = self.manipulated_object.get_pose().p

        # Target pose
        pose = self.generate_random_target_pose(self.randomness_scale)

        self.target_pose = pose

        if self.target_object is not None:
            self.target_object.set_pose(pose)

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
    env = LabFunctionalEnv(object_category="partnet",
                           object_name="3635",
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
