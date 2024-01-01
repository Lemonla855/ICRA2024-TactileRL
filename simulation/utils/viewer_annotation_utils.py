import time

import sapien.core as sapien
from sapien.utils import Viewer
import numpy as np
from typing import Dict
from pathlib import Path
import json
import os


class ArticulationScaleAnnotator:
    def __init__(self, scene: sapien.Scene, renderer: sapien.VulkanRenderer):
        self.scene = scene
        self.renderer = renderer
        self.viewer = Viewer(renderer)
        self.viewer.set_scene(scene)

    def annotate_scales(self, path_dict: Dict[str, str], result_path: str):
        """
        Annotate articulation size interactively in the viewer
        Args:
            path_dict: dict with key as articulation name (used for saving) and value as urdf path
            result_path: path to save annotation results
        """
        scale_dict = {}
        if os.path.exists(result_path):
            with open(result_path, "r") as f:
                scale_dict = json.load(f)
                print("Successfully load {} records.".format(len(scale_dict)))
                # print(scale_dict)
        item_list = list(path_dict.items())
        i = max(0, len(item_list) - 1)
        while i < len(item_list):
            name, path = item_list[i]
            print("name = ", name)
            scale = 1 if not scale_dict.__contains__(
                name) else scale_dict[name]
            scale_dict[name] = scale
            loader = self.scene.create_urdf_loader()
            loader.fix_root_link = True
            loader.scale = scale
            art = loader.load(path)
            self.scene.step()
            while not self.viewer.closed:
                self.viewer.render()
                if self.viewer.window.key_down("left"):
                    scale -= 0.02
                    loader.scale = scale
                    self.scene.remove_articulation(art)
                    art = loader.load(path)
                    self.scene.step()
                elif self.viewer.window.key_down("right"):
                    scale += 0.02
                    loader.scale = scale
                    self.scene.remove_articulation(art)
                    art = loader.load(path)
                    self.scene.step()
                elif self.viewer.window.key_down("enter"):
                    time.sleep(0.1)
                    i = i + 1
                    scale_dict[name] = scale
                    self.scene.remove_articulation(art)
                    loader = None
                    self.scene.step()
                    with open(result_path, "w") as f:
                        json.dump(scale_dict, f, indent=2)
                    break
                elif self.viewer.window.key_down('l'):
                    time.sleep(0.1)
                    i = max(i - 1, 0)
                    self.scene.remove_articulation(art)
                    loader = None
                    self.scene.step()
                    break


def main():
    from hand_teleop.env.sim_env.constructor import get_engine_and_renderer, add_default_scene_light
    from hand_teleop.utils.common_robot_utils import load_robot, modify_robot_visual
    engine, renderer = get_engine_and_renderer(use_gui=True)
    scene = engine.create_scene()
    add_default_scene_light(scene, renderer)

    robot = load_robot(scene, "allegro_hand_free")
    modify_robot_visual(robot)
    robot.set_pose(sapien.Pose([-0.55, 0, 0.14]))

    faucet_list = [
        148, 149, 152, 153, 154, 156, 167, 168, 693, 811, 822, 857, 866, 885,
        908, 920, 929, 931, 960, 991, 1011, 1028, 1034, 1052, 1053, 1280, 1288,
        1343, 1370, 1380, 1386, 1401, 1435, 1444, 1466, 1479, 1492, 1528, 1556,
        1596, 1626, 1633, 1646, 1653, 1667, 1668, 1721, 1741, 1788, 1794, 1795,
        1802, 1817, 1823, 1832, 1886, 1896, 1901, 1903, 1925, 1986, 2017, 2054,
        2082, 2083, 2084, 2113, 2140, 2170
    ]

    partnet_mobility_path = "/home/baochen/Document/sapien_data"
    path_dict = dict()
    for index in faucet_list:
        path_dict[str(
            index)] = f"{partnet_mobility_path}/{index}/mobility.urdf"
    current_dir = Path(__file__).parent
    output_path = current_dir.parent.parent / "assets" / "faucet_scale.json"

    annotator = ArticulationScaleAnnotator(scene, renderer)
    annotator.annotate_scales(path_dict, output_path)


if __name__ == '__main__':
    main()
