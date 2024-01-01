from pathlib import Path
import pdb

import torch.nn as nn

#sys.path.append('../')
from hand_env_utils.arg_utils import *
from hand_env_utils.teleop_env import create_extrinsic_env
from evaluation.dagger_eval_extrinsic import evaluation
from tool.compare import plot_compare

from hand_env_utils.wandb_callback import WandbCallback, setup_wandb
from stable_baselines3.common.vec_env.subproc_vec_env import SubprocVecEnv
from stable_baselines3.dagger.dagger import DAGGER

#=====================tactile===========================
from stable_baselines3.common.torch_layers import CombinedTactileGateExtractor, PointNetExtractor, CombinedExtractor, NatureCNN, ResNet18, NatureCNNSEPERATE
from stable_baselines3.common.utils import get_device
from hand_teleop.utils.tactile_utils import init_tactile
from stable_baselines3.common.torch_layers import PointNetStateExtractor
from stable_baselines3.common.base_class import pretrain_load

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


#main/dagger_extrinsic.py --object_name=any_train --exp=pen --object_cat=pen --robot=xarm7_gripper_front --use_tactile=True --use_rgb=False --use_mask=True --crop=True --dagger_tactile=True

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--n', type=int, default=100)
    parser.add_argument('--workers', type=int, default=10)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--ep', type=int, default=10)
    parser.add_argument('--bs', type=int, default=500)
    parser.add_argument('--seed', type=int, default=500)
    parser.add_argument('--iter', type=int, default=510)
    parser.add_argument('--exp', type=str, default="partnet_bottle3_noheight")
    parser.add_argument('--object_name', type=str, default="4084")
    parser.add_argument('--object_cat', default="bottle3", type=str)
    parser.add_argument('--randomness', type=float, default=1.0)
    parser.add_argument('--horizon', type=int, default=200)  # for debug
    #============================= tactile setting ================================
    parser.add_argument('--use_tactile', default=False, type=str2bool)
    parser.add_argument('--use_rgb', default=True, type=str2bool)
    parser.add_argument('--use_diff', default=False, type=str2bool)
    parser.add_argument('--use_mask', default=False, type=str2bool)
    parser.add_argument('--use_depth', default=False, type=str2bool)
    parser.add_argument('--add_noise', default=False, type=str2bool)
    parser.add_argument('--reduced_state', default=False, type=str2bool)
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
    parser.add_argument('--train_time', default="0625", type=str)
    parser.add_argument('--ent_coef', default=0, type=float)
    parser.add_argument('--cnn_type', default=0, type=int)
    parser.add_argument('--use_buffer', default=False, type=str2bool)
    parser.add_argument('--use_sde', default=False, type=str2bool)
    parser.add_argument('--use_angle', default=False, type=str2bool)
    parser.add_argument('--std', default=1.0, type=float)
    parser.add_argument('--std_coef', default=0.0, type=float)
    parser.add_argument('--NatureCNN', default=False, type=str2bool)
    parser.add_argument('--task_index', default=2, type=int)
    parser.add_argument('--crop', default=False, type=str2bool)
    parser.add_argument('--dagger_tactile', default=False, type=str2bool)
    parser.add_argument(
        '--expert_path',
        default=
        "results/all_kind/ppo-any_train-2-all_kind_multi_buffer_noise_random_righthanddown-100/model/model_490",
        type=str)
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

    if "down" in args.robot:
        exp = exp + "down"

    if args.use_tactile:

        if args.crop:

            exp_keywords = [
                "dagger", args.object_name,
                str(args.task_index), exp, "tactile_crop",
                str(args.seed)
            ]
        else:

            exp_keywords = [
                "dagger", args.object_name,
                str(args.task_index), exp, "tactile",
                str(args.seed)
            ]
    else:

        exp_keywords = [
            "dagger", args.object_name,
            str(args.task_index), exp,
            str(args.seed)
        ]

    if args.use_depth:
        exp_keywords[-2] += "_depth"

    if args.use_mask:
        exp_keywords[-2] += "_mask"

    exp_name = "-".join(exp_keywords)

    use_visual_obs = False

    result_path = Path("./results") / args.train_time / exp_name
    result_path.mkdir(exist_ok=True, parents=True)

    num_cameras = 2

    def create_env_fn():
        environment = create_extrinsic_env(
            object_name=args.object_name,
            use_visual_obs=use_visual_obs,
            pc_noise=args.noise_pc,
            object_category=args.object_cat,
            use_tactile=True,
            use_mask=True,
            use_depth=False,
            use_rgb=False,
            use_tips=args.use_tips,
            robot=args.robot,
            use_orientation=args.use_orientation,
            regrasp=args.regrasp,
            use_buffer=True,
            task_index=args.task_index,
            use_expert=True,
            dagger_tactile=args.dagger_tactile)

        return environment

    def create_eval_env_fn():
        environment = create_extrinsic_env(
            object_name=args.object_name,
            use_visual_obs=use_visual_obs,
            pc_noise=args.noise_pc,
            is_eval=True,
            object_category=args.object_cat,
            use_tactile=True,
            use_mask=True,
            use_depth=False,
            use_rgb=False,
            use_tips=args.use_tips,
            robot=args.robot,
            use_orientation=args.use_orientation,
            regrasp=args.regrasp,
            use_buffer=True,
            task_index=args.task_index,
            use_expert=True,
            dagger_tactile=args.dagger_tactile)

        return environment

    def create_expert_env_fn():
        environment = create_extrinsic_env(
            object_name=args.object_name,
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
            use_expert=False)

        return environment

    #=====================tactile===========================
    ###         init related information for the pytorch3d
    #=====================tactile===========================
    ''' init related information for the pytorch3d'''
    tactile_kwargs = dict(globalScalings=1,
                          object_category=args.object_cat,
                          obj_name=args.object_name,
                          device=get_device(),
                          num_cameras=num_cameras,
                          use_mask=True,
                          use_depth=False,
                          use_rgb=False,
                          use_diff=args.use_diff,
                          add_noise=args.add_noise,
                          image_size=args.image_size,
                          optical_flow=args.optical_flow,
                          task_index=1,
                          crop=args.crop,
                          use_expert=True,
                          random=args.random)

    #=====================tactile===========================
    expert_env = create_expert_env_fn()

    env = SubprocVecEnv([create_env_fn] * args.workers,
                        "spawn",
                        expert_observation_space=expert_env.observation_space,
                        dagger_tactile=args.dagger_tactile,
                        **tactile_kwargs)

    print(env)
    print(env.observation_space, env.action_space)

    wandb_run = setup_wandb(config,
                            exp_name,
                            tags=["state", "relocate", args.object_name],
                            project="extrinsic")

    if args.dagger_tactile:

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
            "features_extractor_class": NatureCNN,  #NatureCNN,
            "cnn_output_dim": args.CNN,
            "state_key": state_key,
            "augmentation": args.random_shift,
            "cnn_type": args.cnn_type,
            "dagger_tactile": args.dagger_tactile
        }
        policy_kwargs = {
            "features_extractor_class": feature_extractor_class,
            "features_extractor_kwargs": feature_extractor_kwargs,
            "net_arch": [dict(pi=[64, 64], vf=[64, 64])],
            "activation_fn": nn.ReLU,
            "log_std_init": np.log(args.std)
        }
    else:
        policy = "MlpPolicy"
        policy_kwargs = {
            'activation_fn': nn.ReLU,
            "log_std_init": np.log(args.std)
        }

    from stable_baselines3.ppo import PPO

    expert = PPO.load(args.expert_path, expert_env, device=get_device("auto"))

    if args.dagger_tactile:
        dagger_observation_space = env.observation_space
    else:
        dagger_observation_space = expert_env.observation_space

    model = DAGGER(
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
        use_sde=args.use_sde,
        expert=expert,
        expert_observation_space=dagger_observation_space,
        use_tactile=args.dagger_tactile)

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
                               dagger_tactile=args.dagger_tactile,
                               **tactile_kwargs),
    )
    evaluation(args, exp_name)

    wandb_run.finish()
