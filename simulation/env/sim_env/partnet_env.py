import pdb
import numpy as np
import sapien.core as sapien
import transforms3d

from hand_teleop.env.sim_env.base import BaseSimulationEnv

from hand_teleop.real_world import lab

from hand_teleop.utils.render_scene_utils import set_entity_color

from hand_teleop.utils.partnet_utils import PARTNET_ORIENTATION
from hand_teleop.utils.partnet_class_utils import load_partnet_object, OBJECT_ORIENTATION, BOTTLE_ANYTRAIN, BOX_ANYTRAIN
from hand_teleop.utils.tactile_loader_parnet_utils import save_partnet_mesh
from hand_teleop.utils.bottle4_helper import SCALE_FACTOR, HAND_SCALE_FACTOR
from hand_teleop.utils.bottle4_helper import CYLINDER_ANYTRAIN, CAPSULE_ANYTRAIN, THINCAPSULE2_ANYTRAIN, THINCAPSULE3_ANYTRAIN, CONE_ANYTRAIN, ICOSPHERE_ANYTRAIN, TORUS_ANYTRAIN, UPTORUS_ANYTRAIN, THINCAPSULE_ANYTRAIN, ANY_TRAIN
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
                 random=False,
                 noise=False,
                 regrasp=False,
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
        self.random = random
        self.noise = noise
        self.regrasp = regrasp

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
        self.manipulated_object.set_name("target_object")
        #===========================tactile============================
        self.obj_index = 0
        self.init_obj_pose = [0, 0, 0]
        #===========================tactile============================

    def load_object(self, object_name, random_size=None):

        if "any" in self.object_name:

            if self.object_category in ["bottle", "bottle2", "bottle3"]:

                index = int(self.object_name.split("_")[2])
                object_name = self.np_random.choice(BOTTLE_ANYTRAIN[index])

            if self.object_category == "box":
                names = BOX_ANYTRAIN

            if self.object_category in ["bottle4"]:

                if "cylinder" in self.object_name:

                    self.obj_index = int(
                        self.np_random.choice(
                            np.arange(0,
                                      len(CYLINDER_ANYTRAIN) - 1).astype(
                                          np.int)))
                    object_name = CYLINDER_ANYTRAIN[self.obj_index]

                elif "train" in self.object_name:
                    self.obj_index = int(
                        self.np_random.choice(
                            np.arange(0, len(ANY_TRAIN)).astype(np.int)))
                    object_name = ANY_TRAIN[self.obj_index]

                elif "thincapsule" in self.object_name:

                    self.obj_index = int(
                        self.np_random.choice(
                            np.arange(0, len(THINCAPSULE_ANYTRAIN)).astype(
                                np.int)))
                    object_name = THINCAPSULE_ANYTRAIN[self.obj_index]

                elif "thincapsule3" in self.object_name:

                    self.obj_index = int(
                        self.np_random.choice(
                            np.arange(0, len(THINCAPSULE3_ANYTRAIN)).astype(
                                np.int)))
                    object_name = THINCAPSULE3_ANYTRAIN[self.obj_index]

                elif "thincapsule2" in self.object_name:
                    self.obj_index = int(
                        self.np_random.choice(
                            np.arange(0, len(THINCAPSULE2_ANYTRAIN)).astype(
                                np.int)))
                    object_name = THINCAPSULE2_ANYTRAIN[self.obj_index]

                elif "any_capsule" in self.object_name:
                    self.obj_index = int(
                        self.np_random.choice(
                            np.arange(0,
                                      len(CAPSULE_ANYTRAIN)).astype(np.int)))
                    object_name = CAPSULE_ANYTRAIN[self.obj_index]

                elif "cone" in self.object_name:
                    self.obj_index = int(
                        self.np_random.choice(
                            np.arange(0, len(CONE_ANYTRAIN)).astype(np.int)))
                    object_name = CONE_ANYTRAIN[self.obj_index]

                elif "icosphere" in self.object_name:
                    self.obj_index = int(
                        self.np_random.choice(
                            np.arange(0,
                                      len(ICOSPHERE_ANYTRAIN)).astype(np.int)))
                    object_name = ICOSPHERE_ANYTRAIN[self.obj_index]

                elif "torus" in self.object_name and "up" not in self.object_name:

                    self.obj_index = int(
                        self.np_random.choice(
                            np.arange(0, len(TORUS_ANYTRAIN)).astype(np.int)))
                    object_name = TORUS_ANYTRAIN[self.obj_index]

                elif "uptorus" in self.object_name:

                    self.obj_index = int(
                        self.np_random.choice(
                            np.arange(0,
                                      len(UPTORUS_ANYTRAIN)).astype(np.int)))
                    object_name = UPTORUS_ANYTRAIN[self.obj_index]

        if self.object_category in ["bottle3", "bottle4"]:

            if self.regrasp:

                cap_radius = self.np_random.uniform(low=0.07,
                                                    high=0.13,
                                                    size=1)
                cap_heigh = self.np_random.uniform(low=0.02,
                                                   high=0.045,
                                                   size=1)
                bottle_height = self.np_random.uniform(low=0.10,
                                                       high=0.18,
                                                       size=1)
                scale = [cap_radius, cap_heigh, bottle_height]
            else:
                cap_radius = self.np_random.uniform(low=0.03,
                                                    high=0.06,
                                                    size=1)
                # cap_heigh = self.np_random.uniform(low=0.05, high=0.07, size=1)
                # bottle_height = self.np_random.uniform(low=0.15,
                #                                        high=0.18,
                #                                        size=1)
                cap_heigh = self.np_random.uniform(low=0.05, high=0.07, size=1)
                bottle_height = self.np_random.uniform(low=0.10,
                                                       high=0.15,
                                                       size=1)
                scale = [cap_radius, cap_heigh, bottle_height]

        else:
            scale = None

        self.current_object_name = object_name

        if self.object_category in ["bottle4"]:

            if "cylinder" in object_name:
                scale_factor = SCALE_FACTOR["cylinder"]
            elif "cone" in object_name:
                scale_factor = SCALE_FACTOR["cone"]
            elif "icosphere" in object_name:
                scale_factor = SCALE_FACTOR["icosphere"]
            elif "capsule" in object_name:
                scale_factor = SCALE_FACTOR["capsule"]
            elif "torus" in object_name:
                scale_factor = SCALE_FACTOR["torus"]
            elif "up" in object_name:
                scale_factor = SCALE_FACTOR["uptorus"]

            if "down" in self.robot_name:

                if "capsule" in object_name:
                    x_scale_fator = HAND_SCALE_FACTOR["down"]["capsule"]

                elif "torus" in object_name:
                    x_scale_fator = HAND_SCALE_FACTOR["down"]["torus"]

                elif "up" in object_name:
                    x_scale_fator = HAND_SCALE_FACTOR["down"]["uptorus"]

                elif "cylinder" in object_name:
                    x_scale_fator = HAND_SCALE_FACTOR["down"][object_name]

            else:
                x_scale_fator = HAND_SCALE_FACTOR['front']
                scale_factor = scale_factor * 1.6

            # scale = [
            #     cap_radius[0] * x_scale_fator * 1.5,
            #     cap_heigh[0] * scale_factor,
            #     bottle_height[0],
            # ]
            scale = [
                cap_radius[0] * x_scale_fator * 0.6,
                0.08,
                0.15,
            ]
        self.rescale = scale

        if "down" in self.robot_name:
            down = True
        else:
            down = False

        manipulated_object, object_height, self.scale = load_partnet_object(
            self.scene,
            self.object_category,
            model_name=object_name,
            random=self.random,
            scale=scale,
            renderer=self.renderer,
            down=down)

        target_object = None
        # if self.use_visual_obs:
        #     target_object.hide_visual()
        if self.renderer and not self.no_rgb:

            if target_object is not None:
                set_entity_color([target_object], [0, 1, 0, 0.6])

        if self.render_mesh:
            save_partnet_mesh(manipulated_object, self.object_category,
                              str(object_name))

        manipulated_object_link_names = [
            link.get_name() for link in manipulated_object.get_links()
        ]
        render_link_name = np.loadtxt("./assets/partnet-mobility-dataset/" +
                                      self.object_category + "/collision/" +
                                      str(object_name) + '/link.txt',
                                      dtype=str)

        self.render_link = [
            manipulated_object.get_links()[manipulated_object_link_names.index(
                name)] for name in render_link_name[-1:]
        ]

        return manipulated_object, target_object, object_height

    def generate_random_object_pose(self, randomness_scale):
        pos = self.np_random.uniform(low=-0.1, high=0.1,
                                     size=2) * randomness_scale

        pre_orientation = OBJECT_ORIENTATION[self.object_category]

        if self.use_orientation > 0.01:
            x_angles = self.np_random.uniform(
                low=-1, high=1,
                size=1)[0] * self.use_orientation  #* np.pi/ 180
            y_angles = self.np_random.uniform(
                low=-1, high=1,
                size=1)[0] * self.use_orientation  #* np.pi/ 180

            # post_orientation = R.from_rotvec([x_angles * 1, 0, 0])
            post_orientation = R.from_euler('zyx', [0, y_angles, x_angles],
                                            degrees=True)
            # print(post_orientation.as_rotvec())

            r = post_orientation.as_matrix() @ R.from_quat(
                np.array(pre_orientation)[[1, 2, 3, 0]]).as_matrix()

            orientation = np.array(R.from_matrix(r).as_quat())[[3, 0, 1, 2]]
        else:
            orientation = pre_orientation

        noise_height = 0.06

        if self.regrasp:

            regrasp_pair = [[0, 0, 0], [-0.06, 0.16, 0.05], [0.15, 0.25, 0.05],
                            [-0.06, 0.25, 0.05]]

            noise_height = 0.16

        if self.noise:

            if self.regrasp:
                position = np.array([
                    pos[0], pos[1], self.object_height +
                    self.np_random.uniform(regrasp_pair[self.regrasp][0],
                                           regrasp_pair[self.regrasp][1], 1)
                ])

            else:
                position = np.array([
                    pos[0], pos[1], self.object_height +
                    self.np_random.uniform(-0.06, noise_height, 1)
                ])

        else:
            position = np.array([pos[0], pos[1], self.object_height])

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
            orientation = OBJECT_ORIENTATION[self.object_category]

        pose = sapien.Pose(position, orientation)
        return pose

    def reset_env(self):

        if "any" in self.object_name:
            self.scene.remove_articulation(self.manipulated_object)
            if self.target_object is not None:
                self.scene.remove_actor(self.target_object)
            self.manipulated_object, self.target_object, self.object_height = self.load_object(
                self.object_name)

        # pose = self.generate_random_object_pose(self.randomness_scale)
        pose = sapien.Pose([0, 0, 0.2],
                           OBJECT_ORIENTATION[self.object_category])

        self.manipulated_object.set_pose(pose)
        self.manipulated_object.set_qpos(
            np.zeros(len(self.manipulated_object.get_qpos())))
        self.init_obj_pose = self.manipulated_object.get_pose().p

        if self.noise:
            # print(self.init_obj_pose)
            noise = self.np_random.uniform(-1, 1, 3)

            noise = noise * 0 / np.linalg.norm(noise) * self.np_random.uniform(
                0.0, 0.03, 1)

            self.init_obj_pose += noise
            self.deviation = noise

        # self.init_obj_pose = self.manipulated_object.get_pose().p

        # Target pose
        # pose = self.generate_random_target_pose(self.randomness_scale)

        self.target_pose = self.init_obj_pose

        if self.target_object is not None:

            self.target_object.set_pose(pose=sapien.Pose(self.target_pose))

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
                           object_name="101401",
                           use_orientation=True,
                           render_mesh=True)
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
