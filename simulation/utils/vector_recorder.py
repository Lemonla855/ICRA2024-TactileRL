from pathlib import Path
import pdb

import torch.nn as nn

#sys.path.append('../')
from hand_env_utils.arg_utils import *

from hand_teleop.utils.tactile_utils import init_tactile, state2tactile

import imageio
import os
import time
import numpy as np

from pathlib import Path
import pdb
import cv2
import torch.nn as nn

#sys.path.append('../')
from hand_env_utils.arg_utils import *

from hand_teleop.utils.tactile_utils import init_tactile, state2tactile
import os
import time
import numpy as np
from tool.plot_result import plot_all
from tool.video import make_video
from tool.plot_result import recaculate_success


# python tool/vector_recorder.py --name=ppo-4084-partnet_bottle3_larger_noise_random_righthand30-500 --time=1107 --robot=allegro_hand_xarm6_wrist_mounted_face_front_30
def mkdir(path):
    if not os.path.exists(path):
        os.mkdir(path)


class VectorRecorder():

    def __init__(self, env, name, train_time):

        self.name = name
        self.env = env

        self.train_time = train_time
        self.workers = 5
        self.horizon = 200

        self.object_category = name.split("-")[2].split("_")[1]
        # self.object_name = name.split("-")[1]

        if "tactile" in name.split("-"):

            self.kind_path = "tactile"
            self.use_tactile = True
        else:
            self.use_tactile = False
            self.kind_path = "state"

        self.path_index = "right_hand"

        self.object_name = name

        mkdir("./tool/analysis")
        mkdir("./tool/analysis/" + train_time)
        mkdir("./tool/analysis/" + train_time + "/" + self.path_index)
        mkdir("./tool/analysis/" + train_time + "/" + self.path_index + "/" +
              self.object_category)
        mkdir("./tool/analysis/" + train_time + "/" + self.path_index + "/" +
              self.object_category + "/" + self.kind_path)
        mkdir("./tool/analysis/" + train_time + "/" + self.path_index + "/" +
              self.object_category + "/" + self.kind_path + "/" + "/" +
              self.object_name)
        mkdir("./tool/analysis/" + train_time + "/" + self.path_index + "/" +
              self.object_category + "/" + self.kind_path + "/" + "/" +
              self.object_name + "/vec_video")

    def record(self, policy, tactile):

        obs = self.env.reset()
        image_lists = [[] for i in range(self.workers)]
        tactile_image_lists = [[] for i in range(self.workers)]

        for _ in range(self.horizon):

            action, _ = policy.predict(obs)
            obs, reward, done, infos = self.env.step(action)

            for index, info in enumerate(infos):

                image_lists[index].append(
                    (np.array(info["imgs"]) * 255).astype(np.uint8))

            if bool(tactile):

                tactile_image = obs["tactile_image"]
                for i in range(self.workers):
                    tactile_image_lists[i].append(tactile_image[i])

        for index, img_list in enumerate(image_lists):
            writer = imageio.get_writer(
                "./tool/analysis/" + self.train_time + "/" + self.path_index +
                "/" + self.object_category + "/" + self.kind_path + "/" + "/" +
                self.object_name + "/vec_video/video_%d.mp4" % (index),
                fps=40)

            if bool(tactile):
                tactile_writer = imageio.get_writer(
                    "./tool/analysis/" + self.train_time + "/" +
                    self.path_index + "/" + self.object_category + "/" +
                    self.kind_path + "/" + "/" + self.object_name +
                    "/vec_video/tactile_video_%d.mp4" % (index),
                    fps=40)

            for j, img in enumerate(img_list):

                writer.append_data(img)
                tactile_img = tactile_image_lists[index][j]

                tactile_img = np.transpose(tactile_img, (1, 2, 0))
                image_list = [[] for i in range(int(tactile_img.shape[2] / 3))]

                for i in range(int(tactile_img.shape[2] / 3)):
                    image_list[i] = tactile_img[:, :, i * 3:i * 3 + 3]

                color = np.concatenate(image_list, axis=1).astype(np.uint8)

                tactile_writer.append_data(color)

            writer.close()
            tactile_writer.close()

        self.env.close()
