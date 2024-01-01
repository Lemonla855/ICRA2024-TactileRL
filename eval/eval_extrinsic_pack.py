import numpy as np
from hand_teleop.real_world import task_setting

from hand_env_utils.teleop_env import create_extrinsic_env
from hand_teleop.real_world.task_setting import IMG_CONFIG
from hand_teleop.utils.tactile_utils import obtain_tactile_force
from stable_baselines3.dapg import DAPG
from stable_baselines3.ppo import PPO
from hand_teleop.utils.tactile_utils import init_tactile, state2tactile
from stable_baselines3.common.utils import get_device
import cv2
from stable_baselines3.common.torch_layers import RandomShiftsAug
import pdb
import argparse
import os
import json
import pickle
from collections import deque
from hand_teleop.utils.camera_utils import fetch_texture
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import cv2
import open3d as o3d
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

# def pca(img):
#     x, y = np.where(img > 0)
#     if len(x) < 1:
#         return 0, 0

#     y, x = np.where(img > 0)
#     plt.scatter(x, y, s=10)

#     # # x = 64 - x
#     # y = y[::-1]
#     # y = 64 - y
#     data = np.vstack((x, y)).T

#     pca = PCA()
#     pca.fit(data)
#     center = np.mean(data, axis=0)
#     eigenvectors = pca.components_
#     eigenvalues = pca.explained_variance_

#     angle = np.arctan2(eigenvectors[0][1], eigenvectors[0][0])
#     if angle < 0:
#         print(angle / 3.14 * 180 + 180)
#     else:
#         print(angle / 3.14 * 180)
#     if (angle / 3.14 * 180) < -180:
#         angle += 3.14
#     elif (angle / 3.14 * 180) > 180:
#         angle -= 3.14

#     if (450 - env.current_angles - env.ee_angles) < 0:
#         real_angles = 450 - env.current_angles - env.ee_angles
#     else:
#         real_angles = 450 - env.current_angles - env.ee_angles

#     if (angle / 3.14 * 180) < 0 and (angle / 3.14 * 180) > -90:

#         # predicted_angles = abs(angle / 3.14 * 180) + 90
#         # if abs(predicted_angles - real_angles) / predicted_angles > 0.1:
#         #     print(predicted_angles, real_angles, angle / 3.14 * 180,
#         #           abs(predicted_angles - real_angles) / real_angles)

#         return 1, abs(angle / 3.14 * 180) + 90

#     elif (angle / 3.14 * 180) <= 90 and (angle / 3.14 * 180) >= 0:

#         return 2, 180 + (90 - angle / 3.14 * 180)

#     elif (angle / 3.14 * 180) > 90 and (angle / 3.14 * 180) <= 180:

#         return 3, 270 - angle / 3.14 * 180

#     elif (angle / 3.14 * 180) <= -90 and (angle / 3.14 * 180) >= -180:

#         # predicted_angles = (abs(angle / 3.14 * 180) - 90) + 180
#         # if abs(predicted_angles - real_angles) / predicted_angles > 0.1:
#         #     print(predicted_angles, real_angles, angle / 3.14 * 180,
#         #           abs(predicted_angles - real_angles) / real_angles)

#         return 4, (abs(angle / 3.14 * 180) - 90) + 180
#     # else:
#     #     print((angle / 3.14 * 180))
#     #     pdb.set_trace()

#     #     return 5, 0


def pca(img):
    x, y = np.where(img > 0)
    if len(x) < 1:
        return 0, 0, 0

    y, x = np.where(img > 0)
    plt.scatter(x, y, s=10)

    # # x = 64 - x
    # y = y[::-1]
    # y = 64 - y
    data = np.vstack((x, y)).T

    pca = PCA()
    pca.fit(data)
    center = np.mean(data, axis=0)
    eigenvectors = pca.components_
    eigenvalues = pca.explained_variance_

    angle = np.arctan2(eigenvectors[0][1], eigenvectors[0][0])
    if angle < 0:
        final_angle = angle / 3.14 * 180 + 180

    else:
        final_angle = angle / 3.14 * 180

    return 0, 180 - final_angle + 90, angle / 3.14 * 180
    # return 0, final_angle, angle / 3.14 * 180


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def add_txt(pil_image, text, color):

    font = ImageFont.truetype("arial.ttf", 60)

    # Create a PIL drawing object
    draw = ImageDraw.Draw(pil_image)

    # Determine the position to place the text
    text_width, text_height = draw.textsize(text, font)

    lines = text.split('\n')
    line_spacing = 10
    line_height = font.getsize(text)[1] + line_spacing

    # Calculate the position of each line based on line spacing
    x = (image_array.shape[1] - text_width) // 2  # Left margin
    y = int(image_array.shape[0] - 150)  # Top margin
    line_height = font.getsize(text)[1] + line_spacing

    # Draw each line of text on the PIL image

    for line in lines:
        draw.text((x, y), line, font=font, fill=color)

        y += line_height
    # x = (image_array.shape[1] - text_width) // 2
    # y = (image_array.shape[0] - text_height) // 2
    # x = (image_array.shape[1] - text_width) // 2
    # y = 10

    # Convert the image back to the original format
    output_image = np.array(pil_image)
    return output_image


if __name__ == '__main__':

    aug = RandomShiftsAug(4)

    parser = argparse.ArgumentParser()
    parser.add_argument('--use_depth', default=False, type=str2bool)
    parser.add_argument('--use_rgb', default=False, type=str2bool)
    parser.add_argument('--use_mask', default=False, type=str2bool)
    parser.add_argument('--use_tactile', default=False, type=str2bool)
    parser.add_argument('--task_index', default=2, type=int)
    parser.add_argument('--use_pc', type=str, default=False)
    parser.add_argument('--noise_pc', type=str, default=False)
    parser.add_argument('--train_time', type=str, default="0524")
    parser.add_argument('--name', type=str, default="ppo-any_train-2-all_kind_multi_noise_random_righthanddown-tactile_crop_depth-100")
   
    parser.add_argument('--seed', type=int, default="100")


    args = parser.parse_args()

    train_time = args.train_time
    name = args.name

    task_property = name.split("-")[2].split("_")
    checkpoint_path = "./results/" + train_time + "/" + name + "/model/model_360"
    # robot = "xarm6_allegro_left"
    robot = "xarm7_gripper_down"
    use_filter = 0
    use_orientation = 0
    use_depth = args.use_depth
    use_rgb = args.use_rgb
    use_tactile = args.use_depth or args.use_mask or args.use_rgb
    image_size = 3

    workers = 10
    iter = 10
    horizon = 200
    total = iter * workers
    threhold = 2 * np.pi

    object_category = "all_kind"
    object_name = name.split("-")[1]
    test_obj_name = name.split("-")[1]
    visual_tactile = True

    if "buffer" in name:
        use_buffer = True
    else:
        use_buffer = False

    if "size" in name:
        index = name.index("size")
        image_size = int(name[index + 4:index + 6])
    else:
        image_size = 64

    if "frames" in name:
        index = name.index("frames")
        num_frames = int(name[index + 6])
    else:
        num_frames = 1

    if "diff" in name:
        use_diff = True
    else:
        use_diff = False

    if "crop" in name:
        crop = True
    else:
        crop = False

    num_cameras = 2

    object_name = name
    test_obj_name = "any_train"  #101727
    use_visual_obs = False
    if args.use_pc:
        use_visual_obs = True

    env = create_extrinsic_env(
        object_name=test_obj_name,
        object_category=object_category,
        use_gui=False,
        robot=robot,
        use_visual_obs=use_visual_obs,
        pc_noise=args.noise_pc,
        is_eval=True,
        #=====================tactile===========================
        use_buffer=use_buffer,
        use_tactile=use_tactile,
        use_depth=use_depth,
        use_rgb=use_rgb,
        task_index=args.task_index,
        novel=False,render_ssp=False

        #=====================tactile===========================
    )

    device = "cuda:0"

    # policy = PPO.load(checkpoint_path, env, device)

    ''' init related information for the pytorch3d'''
    if use_tactile:
        tactile_kwargs = dict(globalScalings=1,
                              object_category=object_category,
                              obj_name=test_obj_name,
                              num_cameras=num_cameras,
                              device=get_device(),
                              visualize_gui=True,
                              show_depth=True,
                              use_mask=args.use_mask,
                              use_depth=use_depth,
                              use_rgb=use_rgb,
                              use_diff=use_diff,
                              add_noise=False,
                              image_size=image_size,
                              optical_flow=False,
                              keypoints=False,
                              task_index=1,
                              crop=crop,
                              novel=False)
        tactile = init_tactile(**tactile_kwargs)

    if visual_tactile:
        tactile_kwargs = dict(globalScalings=1,
                              object_category=object_category,
                              obj_name=test_obj_name,
                              num_cameras=num_cameras,
                              device=get_device(),
                              visualize_gui=True,
                              show_depth=True,
                              use_mask=args.use_mask,
                              use_depth=False,
                              use_rgb=True,
                              use_diff=use_diff,
                              add_noise=False,
                              image_size=image_size,
                              optical_flow=False,
                              keypoints=False,
                              task_index=1)
        tactile_vis = init_tactile(**tactile_kwargs)

    #=====================tactile===========================

    #=====================demonstration===========================

    # if not os.path.exists("./demonstration/" + args.object_category + "/" +
    #                       args.object_name):
    #     os.mkdir("./demonstration/" + args.object_category + "/" +
    #              args.object_name)

    demo = {}

    #=====================demonstration===========================

    print(env.observation_space)
    if env.use_gui:
        viewer = env.render(mode="human")
        viewer.set_camera_rpy(r=0, p=0, y=-np.pi / 2)

    done = False
    manual_action = False
    action = np.ones(22)

    success = 0
    total = 0
    score = 0
    count = 0
    image_index = 0

    images_frames = deque([], maxlen=num_frames)
    env.seed(0)
    obs = env.reset()

    import imageio
    cam = env.cameras["extrinsic_viz"]
    result_path = Path("./video/" + train_time + "/" + name+"/images/")
    result_path.mkdir(exist_ok=True, parents=True)
    
    # env_scene = np.load("scene.npy",allow_pickle=True)
    env_state = 0

    while True:
        reward_sum = 0
        total += 1
        reward = 0

        #verify the render object as same as the seleted obj
        # print(env.manipulated_object, tactile.obj_names[tactile.obj_index])
        # if count == 0:
        obs = env.reset()

        init_object_pose = env.manipulated_object.get_pose()
        # init_target_object_pose = env.target_object.get_pose()

        # print("manipulated object:", env.manipulated_object.name,
        #       "render object:", tactile.obj_names[tactile.obj_index])

        if use_tactile:
            tactile_image = state2tactile(tactile,
                                          obs["tactile_image"],
                                          eval=True,
                                          reset=True)[0]
            if num_frames > 1:
                for _ in range(num_frames):
                    images_frames.append(tactile_image)
                obs["tactile_image"] = images_frames
            else:

                obs["tactile_image"] = tactile_image

        elif visual_tactile:
            tactile_states = env.get_tactile_state()
            # tactile.reset(tactile_states, eval=True)
        #=====================demonstration===========================
        actions = []
        observations = []
        obj_pose = []
        pca_deviation = []
        
        env_scene = np.load("./video/" + train_time + "/" + name+"/tactile_image/"+str(env_state)+"/scene.npy",allow_pickle=True)
        result_path = Path("./video/" + train_time + "/" + name+"/tactile_image/"+str(env_state)+"/"+"robot_image")
        result_path.mkdir(exist_ok=True, parents=True)
        
        #=====================demonstration===========================

        for i in range(200):
            env.scene.unpack(env_scene[int(i)]) 

          
            env.scene.step()
            # env.render()
            env.scene.update_render()
            cam.take_picture()
            rgba = cam.get_float_texture("Color") 
            robot_image = (rgba * 255).clip(0, 255).astype("uint8")
            # rgb = cam.get_color_rgba()
            # rgb = (rgb * 255).astype(np.uint8)
            # robot_image = fetch_texture(cam, "Color",
            #                             return_torch=False)[:, :, :3]
            # cv2.imshow("image",robot_image)
            # cv2.waitKey(1)
            # pdb.set_trace()
            
          
            cv2.imwrite("./video/" + train_time + "/" + name+"/tactile_image/"+str(env_state)+"/"+"robot_image/%d.png" %i,robot_image)
        env_state+=1
        if env_state==2000:
            break

     
     
        