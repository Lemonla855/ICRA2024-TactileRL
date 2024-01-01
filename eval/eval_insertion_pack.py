import numpy as np
from hand_teleop.real_world import task_setting

from hand_env_utils.teleop_env import create_insertion_env, create_functional_class_env
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


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--use_depth', default=False, type=str2bool)
    parser.add_argument('--use_rgb', default=False, type=str2bool)
    parser.add_argument('--use_mask', default=False, type=str2bool)
    parser.add_argument('--novel', default=False, type=str2bool)


    args = parser.parse_args()

    train_time = "0530"
    name = "ppo-any_train-pen_noise_random-insertion-tactile_crop_mask-400"
    # name = "ppo-any_cylinder-partnet_bottle4_newreward_noise_random_righthanddown_size32_frames3-tactile-500"
    task_property = name.split("-")[2].split("_")
    checkpoint_path = "./results/" + train_time + "/" + name + "/model/model_500"
    # robot = "xarm6_allegro_left"
    robot = "xarm7_gripper_front"
    use_tactile = args.use_depth or args.use_mask or args.use_rgb
    use_filter = 0
    use_orientation = 0
    use_depth = args.use_depth
    use_rgb = args.use_rgb
    image_size = 64

    workers = 10
    iter = 10
    horizon = 200
    total = iter * workers
    threhold = 2 * np.pi

    object_category = "pen"
    object_name = name.split("-")[1]
    test_obj_name = name.split("-")[1]
    visual_tactile = False

    if "lowfric" in name:
        index = name.index("lowfric")
        low_friction = float(name[index + 7])
    else:
        low_friction = 0

    if "angle" in name:
        use_angle = True
    else:
        use_angle = False

    if "buffer" in name:
        use_buffer = True
    else:
        use_buffer = False

    if "ori" in name:
        index = name.index("ori")
        use_orientation = float(name[index + 3])
    else:
        use_orientation = 0

    if "random" in task_property:
        use_random = True
    else:
        use_random = False

    if "noise" in task_property:
        use_noise = True
    else:
        use_noise = False

    if "lefthand" in task_property:
        # object_name = object_name + "_lefthand"
        robot = "xarm6_allegro_left"
        path_index = "left_hand"
    else:
        path_index = "right_hand"
        robot = robot

    if "regrasp" in name:

        index = name.find("regrasp")
        regrasp = int(name[index + 7])

    else:
        regrasp = 0

    if "size" in name:
        index = name.index("size")
        image_size = int(name[index + 4:index + 6])
    else:
        image_size = 64

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
    # test_obj_name = "957f897018c8c387b79156a61ad4c01"

    env = create_insertion_env(
        object_name=test_obj_name,
        object_category=object_category,
        use_visual_obs=False,
        use_gui=False,
        robot=robot,
        #=====================tactile===========================
        use_tactile=use_tactile,
        use_depth=use_depth,
        use_rgb=use_rgb,
        random=use_random,
        eval=False,
        noise=use_noise,
        is_eval=True,
        use_filter=use_filter,
        regrasp=regrasp,
        use_orientation=use_orientation,
        image_size=image_size,
        low_friction=low_friction,
        use_buffer=use_buffer,
        num_cameras=num_cameras,
        use_angle=use_angle,
        reverse=False,
        use_mask=args.use_mask,novel=args.novel

        #=====================tactile===========================
    )

    device = "cuda:0"
    policy = PPO.load(checkpoint_path, env, device)

    # env.manipulated_object.get_collision_shapes()[0].rest_offset = -0.1
    # env.setup_tactile_camera_from_config(task_setting.CAMERA_CONFIG["tactile"])
    #=====================tactile===========================
    ###         init related information for the pytorch3d
    #=====================tactile===========================
    ''' init related information for the pytorch3d'''
    if visual_tactile or use_tactile:
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
                              task_index=1,
                              keypoints=False,
                              crop=crop,novel=args.novel)
        tactile = init_tactile(**tactile_kwargs)

    #=====================tactile===========================

    #=====================demonstration===========================

    # if not os.path.exists("./demonstration/" + args.object_category + "/" +
    #                       args.object_name):
    #     os.mkdir("./demonstration/" + args.object_category + "/" +
    #              args.object_name)

    demo = {}

    #=====================demonstration===========================
    if env.use_gui:
        viewer = env.render(mode="human")
        viewer.set_camera_rpy(r=0, p=0, y=-np.pi/2)  # change the viewer direction

  
    env.seed(200)
 
    import imageio
    cam = env.cameras["insertion_viz"]
    from pathlib import Path
    
    
    # env_scene = np.load("insertion_scene.npy",allow_pickle=True)
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
      

        for i in [0,199]:
            
            env.scene.unpack(env_scene[int(i)])


            
            env.scene.step()
            if env.use_gui:
                env.render()
            env.scene.update_render()
            cam.take_picture()
            rgba = cam.get_float_texture("Color") 
            robot_image = (rgba * 255).clip(0, 255).astype("uint8")
            rgb = cam.get_color_rgba()
            rgb = (rgb * 255).astype(np.uint8)
          
            # cv2.imshow("image",robot_image)
            # cv2.waitKey(1)
            # pdb.set_trace()
            
          
           
            cv2.imwrite("./video/" + train_time + "/" + name+"/tactile_image/"+str(env_state)+"/"+"robot_image/%d.png" %i,robot_image)
         
        env_state+=1
        if env_state==5000:
            break

     
     
        