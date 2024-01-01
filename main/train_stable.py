from pathlib import Path
import pdb

import torch.nn as nn

#sys.path.append('../')
from hand_env_utils.arg_utils import *
from hand_env_utils.teleop_env import create_stable_env

from tool.compare import plot_compare

from hand_env_utils.wandb_callback import WandbCallback, setup_wandb
from stable_baselines3.common.vec_env.subproc_vec_env import SubprocVecEnv
from stable_baselines3.ppo import PPO

#=====================tactile===========================
from stable_baselines3.common.torch_layers import CombinedTactileGateExtractor, PointNetExtractor, CombinedExtractor, NatureCNN, ResNet18, NatureCNNSEPERATE
from stable_baselines3.common.utils import get_device
from hand_teleop.utils.tactile_utils import init_tactile
from stable_baselines3.common.torch_layers import PointNetStateExtractor
from stable_baselines3.common.base_class import pretrain_load

from evaluation.eval_stable import evaluation
import numpy as np


#=====================tactile===========================
def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


#main/train_stable.py --object_name=any_train --exp=breaking --object_cat=breaking --robot=xarm7_gripper_down --use_buffer=True
#main/train_stable.py --object_name=any_train --exp=bottle --object_cat=bottle --robot=xarm7_gripper_down --use_buffer=True
#main/train_stable.py --object_name=any_train --exp=bucket --object_cat=bucket --robot=xarm7_gripper_down --use_buffer=True
#main/train_stable.py --object_name=any_train --exp=USB --object_cat=USB --robot=xarm7_gripper_down --use_buffer=True
#main/train_stable.py --object_name=any_train --exp=pen --object_cat=pen --robot=xarm7_gripper_down --use_buffer=True
#main/train_stable.py --object_name=any_train --exp=mug --object_cat=03797390 --robot=xarm7_gripper_down --use_buffer=True
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--n', type=int, default=100)
    parser.add_argument('--workers', type=int, default=10)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--ep', type=int, default=10)
    parser.add_argument('--bs', type=int, default=500)
    parser.add_argument('--seed', type=int, default=500)
    parser.add_argument('--iter', type=int, default=500)
    parser.add_argument('--exp', type=str, default="partnet_bottle3_noheight")
    parser.add_argument('--object_name', type=str, default="4084")
    parser.add_argument('--object_cat', default="bottle3", type=str)
    parser.add_argument('--randomness', type=float, default=1.0)
    parser.add_argument('--horizon', type=int, default=200)  # for debug
    #============================= tactile setting ================================
    parser.add_argument('--use_tactile', default=False, type=str2bool)
    parser.add_argument('--use_rgb', default=False, type=str2bool)
    parser.add_argument('--use_diff', default=False, type=str2bool)
    parser.add_argument('--use_mask', default=False, type=str2bool)
    parser.add_argument('--use_depth', default=False, type=str2bool)
    parser.add_argument('--add_noise', default=False, type=str2bool)
    parser.add_argument('--reduced_state', default=False, type=str2bool)
    parser.add_argument('--noise_table', default=False, type=str2bool)
    #============================= tactile setting ================================

    #============================= training setting ================================
    parser.add_argument('--pretrained', default=False, type=str2bool)
    parser.add_argument('--pretrained_path', default=None, type=str)
    parser.add_argument('--use_pc', default=False, type=str2bool)
    parser.add_argument('--noise_pc', default=False, type=str2bool)
    parser.add_argument('--use_tips', default=True, type=str2bool)
    parser.add_argument('--use_orientation', default=0, type=float)
    parser.add_argument('--robot', default="xarm6_allegro_sfr_down", type=str)
    parser.add_argument('--reverse', default=False, type=str2bool)
    parser.add_argument('--use_bn', type=bool, default=True)
    #============================= training setting ================================

    #============================= task setting ================================
    parser.add_argument('--regrasp', default=0, type=int)
    parser.add_argument('--random', default=False, type=str2bool)
    parser.add_argument('--noise', default=True, type=str2bool)
    parser.add_argument('--low_friction', default=0, type=int)
    parser.add_argument('--image_layer', default="NatureCNN", type=str)
    parser.add_argument('--optical_flow', default=False, type=str2bool)

    parser.add_argument('--CNN', default=256, type=int)
    parser.add_argument('--image_size', default=64, type=int)
    parser.add_argument('--use_filter', default=0, type=int)
    parser.add_argument('--train_time', default="0718", type=str)
    parser.add_argument('--ent_coef', default=0, type=float)
    parser.add_argument('--cnn_type', default=0, type=int)
    parser.add_argument('--use_buffer', default=False, type=str2bool)
    parser.add_argument('--use_sde', default=False, type=str2bool)
    parser.add_argument('--use_angle', default=False, type=str2bool)
    parser.add_argument('--std', default=1.0, type=float)
    parser.add_argument('--std_coef', default=0.0, type=float)
    parser.add_argument('--NatureCNN', default=False, type=str2bool)
    parser.add_argument('--task_index', default=0, type=int)
    parser.add_argument('--crop', default=True, type=str2bool)
    parser.add_argument(
        '--random_shift',
        default=True,
        type=str2bool,
    )
    #============================= task setting ================================

    args = parser.parse_args()

    horizon = args.horizon
    env_iter = args.iter * horizon * args.n

    config = {
        'n_env_horizon': args.n,
        "object_cat": args.object_cat,
        'object_name': args.object_name,
        'update_iteration': args.iter,
        'total_step': env_iter,
        'randomness': args.randomness,
    }

    exp = args.exp

    if args.low_friction:
        exp = exp + "_lowfric" + str(args.low_friction)

    if args.NatureCNN:
        image_class = NatureCNN
    else:
        image_class = NatureCNNSEPERATE
        exp += "_multi"

    if args.image_layer not in ["NatureCNN"]:
        image_class = ResNet18
        exp = exp + "_ResNet18"
    else:
        image_class = NatureCNN
    image_class = NatureCNN

    if args.use_buffer:
        exp = exp + "_buffer"

    if args.noise:
        exp = exp + "_noise"

    if args.random:
        exp = exp + "_randomimage"

    if "left" in args.robot:
        exp = exp + "_lefthand"
    else:
        exp = exp + "_righthand"
        # if len(args.robot.split("_")) > 7:
        #     exp = exp + args.robot.split("_")[-1]
        if "30" in args.robot:
            exp = exp + "30"

    if "down" in args.robot:
        exp = exp + "down"

    if "sfr" in args.robot:
        exp = exp + "fsr"

    if args.reduced_state:
        exp = exp + "_reduced"

    if args.use_angle:
        exp = exp + "_angle"

    if args.use_tactile and not args.random_shift:
        exp = exp + "_NoAug"

    if args.CNN != 256:
        exp = exp + "_MLP" + str(args.CNN)

    if args.bs != 500:
        exp = exp + "_bs" + str(args.bs)

    if args.cnn_type != 0:
        exp = exp + "_cnntype" + str(args.cnn_type)

    if args.image_size != 64:
        exp = exp + "_size" + str(args.image_size)

    if args.use_sde:
        exp = exp + "_sde"

    if args.std_coef > 0.001:
        exp = exp + "_stdcoef" + str(args.std_coef)

    if args.std < 0.9:
        exp = exp + "_std" + str(args.std)

    if args.regrasp:

        exp = exp + "_regrasp" + str(args.regrasp)

    if args.use_orientation > 0:
        exp = exp + "_ori" + str(args.use_orientation)

    if args.use_filter > 0:
        exp = exp + "_filter" + str(args.use_filter)

    if args.ent_coef > 1e-5:
        exp = exp + "_ent" + str(args.ent_coef)

    if args.reverse:
        exp = exp + "_reverse"

    if args.use_diff:
        exp = exp + "_diff"

    if args.optical_flow:
        exp = exp + "_optical"

    use_visual_obs = False

    if args.use_tactile:

        if args.crop:

            exp_keywords = [
                "ppo_stable", args.object_name,
                str(args.task_index), exp, "tactile_crop",
                str(args.seed)
            ]
        else:

            exp_keywords = [
                "ppo_stable", args.object_name,
                str(args.task_index), exp, "tactile",
                str(args.seed)
            ]

        use_visual_obs = False
    elif args.use_pc:
        if args.noise_table:
            exp_keywords = [
                "ppo_pc_noise_table", args.object_name,
                str(args.task_index), exp,
                str(args.seed)
            ]

        elif not args.noise_pc:
            exp_keywords = [
                "ppo_pc", args.object_name,
                str(args.task_index), exp,
                str(args.seed)
            ]
        else:
            exp_keywords = [
                "ppo_pc_noise", args.object_name,
                str(args.task_index), exp,
                str(args.seed)
            ]

        exp_name = "-".join(exp_keywords)

        result_path = Path("./results") / args.train_time
        result_path.mkdir(exist_ok=True, parents=True)

        result_path = Path("./results") / args.train_time / exp_name
        result_path.mkdir(exist_ok=True, parents=True)

        use_visual_obs = True
    else:

        exp_keywords = [
            "ppo_stable", args.object_name,
            str(args.task_index), exp,
            str(args.seed)
        ]
        use_visual_obs = False

    if args.use_depth:
        exp_keywords[-2] += "_depth"

    if args.use_mask:
        exp_keywords[-2] += "_mask"

    exp_name = "-".join(exp_keywords)

    result_path = Path("./results") / args.train_time
    result_path.mkdir(exist_ok=True, parents=True)

    result_path = Path("./results") / args.train_time / exp_name
    result_path.mkdir(exist_ok=True, parents=True)

    num_cameras = 2

    def create_env_fn():
        environment = create_stable_env(object_name=args.object_name,
                                        use_visual_obs=use_visual_obs,
                                        pc_noise=args.noise_pc,
                                        object_category=args.object_cat,
                                        use_tactile=args.use_tactile,
                                        use_mask=args.use_mask,
                                        use_depth=args.use_depth,
                                        use_rgb=args.use_rgb,
                                        use_tips=args.use_tips,
                                        robot=args.robot,
                                        use_orientation=args.use_orientation,
                                        regrasp=args.regrasp,
                                        use_buffer=args.use_buffer,
                                        task_index=args.task_index,
                                        noise_table=args.noise_table)

        return environment

    def create_eval_env_fn():
        environment = create_stable_env(object_name=args.object_name,
                                        object_category=args.object_cat,
                                        use_visual_obs=use_visual_obs,
                                        pc_noise=args.noise_pc,
                                        is_eval=True,
                                        use_tactile=args.use_tactile,
                                        use_mask=args.use_mask,
                                        use_depth=args.use_depth,
                                        use_rgb=args.use_rgb,
                                        use_tips=args.use_tips,
                                        robot=args.robot,
                                        use_orientation=args.use_orientation,
                                        regrasp=args.regrasp,
                                        use_buffer=args.use_buffer,
                                        task_index=args.task_index,
                                        noise_table=args.noise_table)
        return environment

    #=====================tactile===========================
    ###         init related information for the pytorch3d
    #=====================tactile===========================
    if args.use_tactile:
        ''' init related information for the pytorch3d'''
        tactile_kwargs = dict(globalScalings=1,
                              object_category=args.object_cat,
                              obj_name=args.object_name,
                              device=get_device(),
                              num_cameras=num_cameras,
                              use_mask=args.use_mask,
                              use_depth=args.use_depth,
                              use_rgb=args.use_rgb,
                              use_diff=args.use_diff,
                              add_noise=args.add_noise,
                              image_size=args.image_size,
                              optical_flow=args.optical_flow,
                              task_index=1,
                              crop=args.crop,
                              random=args.random)
    else:
        tactile_kwargs = dict()
    #=====================tactile===========================

    env = SubprocVecEnv([create_env_fn] * args.workers, "spawn",
                        **tactile_kwargs)
    print(env)
    print(env.observation_space, env.action_space)

    wandb_run = setup_wandb(config,
                            exp_name,
                            tags=["state", "relocate", args.object_name],
                            project="stable_place")

    if args.use_tactile:

        if args.use_pc:
            state_key = "state"  #pc use robot state
        else:
            # if not args.regrasp:
            #     state_key = "oracle_state"  #state use oracle state
            # else:
            #     state_key = "state"  #state use oracle state
            state_key = "oracle_state"

        policy = "MultiInputPolicy"

        feature_extractor_class = CombinedExtractor

        feature_extractor_kwargs = {
            "key": "tactile_image",
            "features_extractor_class": image_class,  #NatureCNN,
            "cnn_output_dim": args.CNN,
            "state_key": state_key,
            "augmentation": args.random_shift,
            "cnn_type": args.cnn_type,
        }
        policy_kwargs = {
            "features_extractor_class": feature_extractor_class,
            "features_extractor_kwargs": feature_extractor_kwargs,
            "net_arch": [dict(pi=[64, 64], vf=[64, 64])],
            "activation_fn": nn.ReLU,
            "log_std_init": np.log(args.std)
        }

    elif use_visual_obs:
        policy = "PointCloudPolicy"

        feature_extractor_class = PointNetStateExtractor
        feature_extractor_kwargs = {
            "pc_key": "extrinsic-point_cloud",
            "local_channels": (64, 128, 256),
            "global_channels": (256, ),
            "use_bn": args.use_bn,
            "state_mlp_size": (64, 64),
        }
        policy_kwargs = {
            "features_extractor_class": feature_extractor_class,
            "features_extractor_kwargs": feature_extractor_kwargs,
            "net_arch": [dict(pi=[64, 64], vf=[64, 64])],
            "activation_fn": nn.ReLU,
        }

    else:
        policy = "MlpPolicy"
        policy_kwargs = {
            'activation_fn': nn.ReLU,
            "log_std_init": np.log(args.std)
        }

    model = PPO(
        policy,
        env,
        verbose=1,
        n_epochs=args.ep,
        n_steps=(args.n // args.workers) * horizon,
        learning_rate=args.lr,
        batch_size=args.bs,
        seed=args.seed,
        policy_kwargs=policy_kwargs,
        tensorboard_log=None,  #str(result_path / "log"),
        min_lr=args.lr,
        max_lr=args.lr,
        adaptive_kl=0.02,
        target_kl=0.2,
        std_coef=args.std_coef,
        ent_coef=args.ent_coef,
        use_sde=args.use_sde)

    if args.pretrained:

        if args.pretrained_path is not None:

            pretrain_load(args.pretrained_path,
                          get_device(),
                          pretrain_model=model)
            # model.load(args.pretrained_path, env, get_device())
        else:
            NotImplementedError

    model.learn(
        total_timesteps=int(env_iter),
        callback=WandbCallback(model_save_freq=10,
                               model_save_path=str(result_path / "model"),
                               eval_env_fn=create_eval_env_fn,
                               eval_freq=10,
                               eval_cam_names=["relocate_viz"],
                               exp_name=None,
                               train_env=None,
                               train_time=None,
                               viz_point_cloud=use_visual_obs,
                               **tactile_kwargs),
    )
    evaluation(args, exp_name)
    wandb_run.finish()
