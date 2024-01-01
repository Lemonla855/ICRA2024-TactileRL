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
    parser.add_argument('--object_cat', type=str, default="all_kind")
    parser.add_argument('--train_time', type=str, default="all_kind")
    parser.add_argument('--name', type=str, default="ppo-any_train-2-all_kind_multi_noise_random_righthanddown-tactile_crop_mask-500")
    parser.add_argument('--itr', type=int, default="400")
    parser.add_argument('--seed', type=int, default="500")



    args = parser.parse_args()

    train_time = args.train_time
    name = args.name

    task_property = name.split("-")[2].split("_")
    checkpoint_path = "./results/" + train_time + "/" + name + "/model/model_%d" %args.itr
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

    object_category = args.object_cat
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
        novel=False

        #=====================tactile===========================
    )

    device = "cuda:0"

    policy = PPO.load(checkpoint_path, env, device)

    # env.manipulated_object.get_collision_shapes()[0].rest_offset = -0.1
    # env.setup_tactile_camera_from_config(task_setting.CAMERA_CONFIG["tactile"])
    #=====================tactile===========================c
    ###         init related information for the pytorch3d
    #=====================tactile===========================
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
                              novel=False,show_gui=False)
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
                              task_index=1,show_gui=True)
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
    env.seed(args.seed)
    obs = env.reset()

    import imageio
    cam = env.cameras["extrinsic_viz"]
    result_path = Path("./video/" + train_time + "/" + name)
    result_path.mkdir(exist_ok=True, parents=True)
    result_path = Path("./video/" + train_time + "/" + name +'/tactile_image')
    result_path.mkdir(exist_ok=True, parents=True)

    result_path = Path("./tactile_image/" + train_time + "/" + name +'/tactile_image')
    result_path.mkdir(exist_ok=True, parents=True)
    robot_video = imageio.get_writer("./video/" + train_time + "/" + name +
                                     "/video.mp4",
                                     fps=30)
    
    env_state_index = 0

    # while not viewer.closed:
    while env_state_index<200:
      
        reward_sum = 0
        total += 1
        reward = 0

        #verify the render object as same as the seleted obj
        # print(env.manipulated_object, tactile.obj_names[tactile.obj_index])
        # if count == 0:
        obs = env.reset()
        
        env_scene = []
        # result_path = Path("./video/" + train_time + "/" + name +'/tactile_image/'+str(env_state_index))
        # result_path.mkdir(exist_ok=True, parents=True)

        result_path = Path("./video/" + train_time + "/" + name +'/tactile_image/'+str(env_state_index))
        result_path.mkdir(exist_ok=True, parents=True)
        result_path = Path("./video/" + train_time + "/" + name +'/tactile_image/'+str(env_state_index)+"/robot_image")
        result_path.mkdir(exist_ok=True, parents=True)


        # Change the visual of object using material
        import sapien.core as sapien
        color = np.array([208, 245, 190, 255]) / 255
        manipulation_object: sapien.Actor = env.manipulated_object
        for visual_body in manipulation_object.get_visual_bodies():
            for render_shape in visual_body.get_render_shapes():
                mat = render_shape.material
                # mat.set_base_color(color)
                mat.set_metallic(0.1)
                mat.set_specular(0.8)
                mat.set_roughness(0.3)
                render_shape.set_material(mat)

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

        #=====================demonstration===========================

        for ho in range(env.horizon):

            # if count == 0:
            if manual_action:
                action = np.zeros(13)

            else:
                # if ho > 0:
                #     obs[-1] = (env.target_angles -
                #                predicted_angles) / 180 * np.pi

                action = policy.predict(observation=obs, deterministic=True)[0]

            #=====================demonstration===========================
            # observations.append(obs.tolist())  #_last_obs, actions
            # actions.append(action.tolist())

            #=====================demonstration===========================

            obs, reward, done, info = env.step(action)
            # obj_pose.append(([
            #     env.manipulated_object.get_pose().p,
            #     env.manipulated_object.get_pose().q
            # ]))

            reward_sum += reward
            if env.use_gui:
                env.render()
            env_scene.append(env.scene.pack())
            
            

            # print(obtain_tactile_force(env.scene,[env.manipulated_object.get_links()[-1]],env.finger_tip_links))
            #=====================tactile===========================
            if use_tactile:

                tactile_image = state2tactile(tactile,
                                              obs["tactile_image"],
                                              eval=True)[0]

                # if num_frames > 1:

                #     images_frames.append(tactile_image)
                #     obs["tactile_image"] = images_frames
                # else:

                obs["tactile_image"] = tactile_image

                # imgs = cv2.imread(
                #     "/media/lme/data/digit_sim2real/data/0508/pen2/raw/%d.png"
                #     % ho)

                # obs["tactile_image"] = np.concatenate(
                #     [imgs[None, 192:, :64, 0], imgs[None, 192:, 64:, 0]],
                #     axis=0)

                # tactile_image = obs["tactile_image"].astype(np.float32)

                # print(np.unique(tactile_image), tactile_image.dtype)

                # env.update_tactile(tactile_vis)
                images = np.transpose((tactile_image / 255), (1, 2, 0))

                if args.use_rgb:
                    image_list = [[] for i in range(int(images.shape[2] / 3))]
                    for i in range(int(images.shape[2] / 3)):
                        image_list[i] = images[:, :, i * 3:i * 3 + 3]

                else:
                    image_list = [[] for i in range(int(images.shape[2] / 1))]
                    for i in range(int(images.shape[2] / 1)):
                        image_list[i] = images[:, :, i * 1:i * 1 + 1]
                color = np.concatenate(image_list, axis=1)

                pred_id, predicted_angles, syn_angle = pca(color[:, :64, 0])
                if (450 - env.current_angles - env.ee_angles) < 0:
                    real_angles = 450 - env.current_angles - env.ee_angles
                else:
                    real_angles = 450 - env.current_angles - env.ee_angles
                # print(predicted_angles, real_angles,
                #       abs(predicted_angles - real_angles) / real_angles)

                # pca_deviation.append(
                #     np.clip(
                #         abs(predicted_angles - real_angles) / real_angles, 0,
                #         1))

                # # if abs(predicted_angles -
                # #        real_angles) / predicted_angles > 0.1:
                # # print(pred_id, predicted_angles, real_angles,
                # #       abs(predicted_angles - real_angles) / real_angles)

                # # if predicted_angles < 0.1:
                # #     print(pred_id, predicted_angles)
                # color = cv2.resize(color,
                #                    (color.shape[1] * 2, color.shape[0] * 2))

                # cv2.imshow("color", cv2.cvtColor(color, cv2.COLOR_RGB2BGR))
                # # print(np.unique(cv2.cvtColor(color, cv2.COLOR_RGB2BGR)))
                # cv2.waitKey(1)
                # print(ho)
                # cv2.imwrite(
                #     "trash/" + str(real_angles) + "_" + str(predicted_angles) +
                #     ".png",
                #     np.array(cv2.cvtColor(color, cv2.COLOR_RGB2BGR) *
                #              255).astype(np.uint8))
                color_n_depth = env.update_tactile(tactile_vis)
                
                cv2.imwrite( "./tactile_image/" + train_time + "/" + name +'/tactile_image/'+str(env_state_index)+'/%d.png' %ho,np.array(cv2.cvtColor(color_n_depth, cv2.COLOR_RGB2BGR)*255).astype(np.uint8))
                
            elif visual_tactile:
                color_n_depth = env.update_tactile(tactile_vis)

                # pred_id, predicted_angles = pca(color_n_depth[64:, 64:, 0])

                # if (450 - env.current_angles - env.ee_angles) < 0:
                #     real_angles = 450 - env.current_angles - env.ee_angles
                # else:
                #     real_angles = 450 - env.current_angles - env.ee_angles

                # pca_deviation.append(
                #     np.clip(
                #         abs(predicted_angles - real_angles) / real_angles, 0,
                #         1))
                # color = np.array(color_n_depth[64:, 64:, 0])
                # dev = abs(predicted_angles - real_angles) / real_angles
                # print(predicted_angles, real_angles)
                # cv2.imwrite(
                #     "trash/" + str(real_angles) + "_" + str(predicted_angles) +
                #     ".png",
                #     np.array(cv2.cvtColor(color, cv2.COLOR_RGB2BGR) *
                #              255).astype(np.uint8))

            #=====================tactile===========================
            env.scene.update_render()
            cam.take_picture()
            # robot_image = fetch_texture(cam, "Color",
            #                             return_torch=False)[:, :, :3]
            rgba = cam.get_float_texture("Color")
            robot_image = (rgba * 255).clip(0, 255).astype("uint8")
            cv2.imwrite("./video/" + train_time + "/" + name+"/tactile_image/"+str(env_state_index)+"/"+"robot_image/%d.png" %ho,robot_image)
       


        if abs(env.error_percentage) < 0.1:
            success += 1
        
        np.save("./video/" + train_time + "/" + name +'/tactile_image/'+str(env_state_index)+"/scene.npy",env_scene)
        np.save("./video/" + train_time + "/" + name +'/tactile_image/'+str(env_state_index)+"/reward.npy",np.array([env.first_theta,env.target_angles,env.current_angles,env.error_percentage]))
        env_state_index+=1
        print(f"Reward: {reward_sum}", [env.first_theta,env.target_angles,env.current_angles,env.error_percentage],
              success / total)
  
        
    # robot_video.close()
