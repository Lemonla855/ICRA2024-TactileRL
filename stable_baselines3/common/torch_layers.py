from itertools import zip_longest
from typing import Dict, List, Tuple, Type, Union, Optional

import gym
import torch
import torch as th
from torch import nn
import torch.nn.functional as F

from stable_baselines3.common.preprocessing import get_flattened_obs_dim, is_image_space
from stable_baselines3.common.type_aliases import TensorDict
from stable_baselines3.common.utils import get_device
from stable_baselines3.networks.pointnet_modules.pointnet import PointNet
#=====================tactile===========================
import numpy as np
from stable_baselines3.common.encoder import PixelEncoder, make_encoder
from torchvision.models import vgg19_bn, vgg16_bn, inception_v3, alexnet, resnet18, resnet34, resnet50
import pdb
#=====================tactile===========================


class BaseFeaturesExtractor(nn.Module):
    """
    Base class that represents a features extractor.
    :param observation_space:
    :param features_dim: Number of features extracted.
    """

    def __init__(self, observation_space: gym.Space, features_dim: int = 0):
        super().__init__()
        assert features_dim > 0
        self._observation_space = observation_space
        self._features_dim = features_dim

    @property
    def features_dim(self) -> int:
        return self._features_dim

    def forward(self, observations: th.Tensor) -> th.Tensor:
        raise NotImplementedError()


class FlattenExtractor(BaseFeaturesExtractor):
    """
    Feature extract that flatten the input.
    Used as a placeholder when feature extraction is not needed.
    :param observation_space:
    """

    def __init__(self, observation_space: gym.Space):
        super().__init__(observation_space,
                         get_flattened_obs_dim(observation_space))
        self.flatten = nn.Flatten()

    def forward(self, observations: th.Tensor) -> th.Tensor:
        return self.flatten(observations)


def weight_init(m):
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0.0)
    elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        gain = nn.init.calculate_gain('relu')
        nn.init.orthogonal_(m.weight.data, gain)
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0.0)


class Drvq2Encoder(BaseFeaturesExtractor):

    def __init__(self,
                 observation_space: gym.spaces.Box,
                 features_dim: int = 512):

        super().__init__(observation_space, features_dim)
        # We assume CxHxW images (channels first)
        # Re-ordering will be done by pre-preprocessing or wrapper
        assert is_image_space(observation_space, check_channels=False), (
            "You should use NatureCNN "
            f"only with images not with {observation_space}\n"
            "(you are probably using `CnnPolicy` instead of `MlpPolicy` or `MultiInputPolicy`)\n"
            "If you are using a custom environment,\n"
            "please check it using our env checker:\n"
            "https://stable-baselines3.readthedocs.io/en/master/common/env_checker.html"
        )
        n_input_channels = observation_space.shape[0]

        self.convnet = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, 3, stride=2), nn.ReLU(),
            nn.Conv2d(32, 32, 3, stride=1), nn.ReLU(),
            nn.Conv2d(32, 32, 3, stride=1), nn.ReLU(),
            nn.Conv2d(32, 32, 3, stride=1), nn.ReLU(), nn.Flatten())

        # Compute shape by doing one forward pass
        with th.no_grad():
            n_flatten = self.convnet(
                th.as_tensor(
                    observation_space.sample()[None]).float()).shape[1]

        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim),
                                    nn.ReLU())

    def forward(self, obs):
        obs = obs - 0.5

        h = self.linear(self.convnet(obs))

        return h


class NatureCNN(BaseFeaturesExtractor):
    """
    CNN from DQN nature paper:
        Mnih, Volodymyr, et al.
        "Human-level control through deep reinforcement learning."
        Nature 518.7540 (2015): 529-533.
    :param observation_space:
    :param features_dim: Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """

    def __init__(self,
                 observation_space: gym.spaces.Box,
                 features_dim: int = 512,
                 cnn_type: int = 0):
        super().__init__(observation_space, features_dim)
        # We assume CxHxW images (channels first)
        # Re-ordering will be done by pre-preprocessing or wrapper
        assert is_image_space(observation_space, check_channels=False), (
            "You should use NatureCNN "
            f"only with images not with {observation_space}\n"
            "(you are probably using `CnnPolicy` instead of `MlpPolicy` or `MultiInputPolicy`)\n"
            "If you are using a custom environment,\n"
            "please check it using our env checker:\n"
            "https://stable-baselines3.readthedocs.io/en/master/common/env_checker.html"
        )
        n_input_channels = observation_space.shape[0]

        if cnn_type == 0:
            self.cnn = nn.Sequential(
                nn.Conv2d(n_input_channels,
                          32,
                          kernel_size=8,
                          stride=4,
                          padding=0),
                nn.ReLU(),
                nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
                nn.ReLU(),
                nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
                nn.ReLU(),
                nn.Flatten(),
            )

        if cnn_type == 1:
            self.cnn = nn.Sequential(
                nn.Conv2d(n_input_channels,
                          32,
                          kernel_size=8,
                          stride=4,
                          padding=0),
                nn.ReLU(),
                nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
                nn.ReLU(),
                # nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
                # nn.ReLU(),
                nn.Flatten(),
            )
        if cnn_type == 2:
            self.cnn = nn.Sequential(
                nn.Conv2d(n_input_channels,
                          16,
                          kernel_size=8,
                          stride=4,
                          padding=0),
                nn.ReLU(),
                # nn.MaxPool2d(kernel_size=2),
                nn.Conv2d(16, 32, kernel_size=4, stride=2, padding=0),
                nn.ReLU(),
                # nn.MaxPool2d(kernel_size=2),
                # nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
                # nn.ReLU(),
                nn.Flatten(),
            )

        if cnn_type == 3:
            self.cnn = nn.Sequential(
                nn.Conv2d(n_input_channels, 32, 3, stride=2), nn.ReLU(),
                nn.Conv2d(32, 32, 3, stride=1), nn.ReLU(),
                nn.Conv2d(32, 32, 3, stride=1), nn.ReLU(),
                nn.Conv2d(32, 32, 3, stride=1), nn.ReLU(), nn.Flatten())

        # Compute shape by doing one forward pass

        with th.no_grad():
            n_flatten = self.cnn(
                th.as_tensor(
                    observation_space.sample()[None]).float()).shape[1]

        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim),
                                    nn.ReLU())

    def forward(self, observations: th.Tensor) -> th.Tensor:

        return self.linear(self.cnn(observations))


class NatureCNNSEPERATE(BaseFeaturesExtractor):
    """
    CNN from DQN nature paper:
        Mnih, Volodymyr, et al.
        "Human-level control through deep reinforcement learning."
        Nature 518.7540 (2015): 529-533.
    :param observation_space:
    :param features_dim: Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """

    def __init__(self,
                 observation_space: gym.spaces.Box,
                 features_dim: int = 512,
                 cnn_type: int = 0):
        super().__init__(observation_space, features_dim)
        # We assume CxHxW images (channels first)
        # Re-ordering will be done by pre-preprocessing or wrapper
        assert is_image_space(observation_space, check_channels=False), (
            "You should use NatureCNN "
            f"only with images not with {observation_space}\n"
            "(you are probably using `CnnPolicy` instead of `MlpPolicy` or `MultiInputPolicy`)\n"
            "If you are using a custom environment,\n"
            "please check it using our env checker:\n"
            "https://stable-baselines3.readthedocs.io/en/master/common/env_checker.html"
        )
        n_input_channels = int(observation_space.shape[0] / 2)
        self.n_input_channels = n_input_channels
        observation_space = gym.spaces.Box(low=0,
                                           high=255,
                                           dtype=np.uint8,
                                           shape=(n_input_channels,
                                                  observation_space.shape[1],
                                                  observation_space.shape[2]))

        if cnn_type == 0:
            self.cnn = nn.Sequential(
                nn.Conv2d(n_input_channels,
                          32,
                          kernel_size=8,
                          stride=4,
                          padding=0),
                nn.ReLU(),
                nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
                nn.ReLU(),
                nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
                nn.ReLU(),
                nn.Flatten(),
            )

        if cnn_type == 1:
            self.cnn = nn.Sequential(
                nn.Conv2d(n_input_channels,
                          32,
                          kernel_size=8,
                          stride=4,
                          padding=0),
                nn.ReLU(),
                nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
                nn.ReLU(),
                # nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
                # nn.ReLU(),
                nn.Flatten(),
            )
        if cnn_type == 2:
            self.cnn = nn.Sequential(
                nn.Conv2d(n_input_channels,
                          16,
                          kernel_size=8,
                          stride=4,
                          padding=0),
                nn.ReLU(),
                # nn.MaxPool2d(kernel_size=2),
                nn.Conv2d(16, 32, kernel_size=4, stride=2, padding=0),
                nn.ReLU(),
                # nn.MaxPool2d(kernel_size=2),
                # nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
                # nn.ReLU(),
                nn.Flatten(),
            )

        if cnn_type == 3:
            self.cnn = nn.Sequential(
                nn.Conv2d(n_input_channels, 32, 3, stride=2), nn.ReLU(),
                nn.Conv2d(32, 32, 3, stride=1), nn.ReLU(),
                nn.Conv2d(32, 32, 3, stride=1), nn.ReLU(),
                nn.Conv2d(32, 32, 3, stride=1), nn.ReLU(), nn.Flatten())

        # Compute shape by doing one forward pass

        with th.no_grad():
            n_flatten = self.cnn(
                th.as_tensor(
                    observation_space.sample()[None]).float()).shape[1]

        self.linear = nn.Sequential(
            nn.Linear(n_flatten, int(features_dim / 2)), nn.ReLU())

    def forward(self, observations: th.Tensor) -> th.Tensor:

        feature1 = self.linear(
            self.cnn(observations[:, :self.n_input_channels]))
        feature2 = self.linear(
            self.cnn(observations[:, self.n_input_channels:]))

        return torch.cat((feature1, feature2), dim=-1)


class ResNet18(BaseFeaturesExtractor):
    """
    CNN from DQN nature paper:
        Mnih, Volodymyr, et al.
        "Human-level control through deep reinforcement learning."
        Nature 518.7540 (2015): 529-533.
    :param observation_space:
    :param features_dim: Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """

    def __init__(self,
                 observation_space: gym.spaces.Box,
                 features_dim: int = 512,
                 cnn_type: int = 0):
        super().__init__(observation_space, features_dim)
        # We assume CxHxW images (channels first)
        # Re-ordering will be done by pre-preprocessing or wrapper
        assert is_image_space(observation_space, check_channels=False), (
            "You should use NatureCNN "
            f"only with images not with {observation_space}\n"
            "(you are probably using `CnnPolicy` instead of `MlpPolicy` or `MultiInputPolicy`)\n"
            "If you are using a custom environment,\n"
            "please check it using our env checker:\n"
            "https://stable-baselines3.readthedocs.io/en/master/common/env_checker.html"
        )
        n_input_channels = observation_space.shape[0]
        self.cnn = resnet18(pretrained=True)
        self.cnn.conv1 = nn.Conv2d(n_input_channels,
                                   64,
                                   kernel_size=(7, 7),
                                   stride=(2, 2),
                                   padding=(3, 3),
                                   bias=False)

        self.cnn.fc = nn.Linear(self.cnn.fc.in_features, features_dim)

    def forward(self, observations: th.Tensor) -> th.Tensor:

        return self.cnn(observations)


def create_mlp(
    input_dim: int,
    output_dim: int,
    net_arch: List[int],
    activation_fn: Type[nn.Module] = nn.ReLU,
    squash_output: bool = False,
) -> List[nn.Module]:
    """
    Create a multi layer perceptron (MLP), which is
    a collection of fully-connected layers each followed by an activation function.
    :param input_dim: Dimension of the input vector
    :param output_dim:
    :param net_arch: Architecture of the neural net
        It represents the number of units per layer.
        The length of this list is the number of layers.
    :param activation_fn: The activation function
        to use after each layer.
    :param squash_output: Whether to squash the output using a Tanh
        activation function
    :return:
    """

    if len(net_arch) > 0:
        modules = [nn.Linear(input_dim, net_arch[0]), activation_fn()]
    else:
        modules = []

    for idx in range(len(net_arch) - 1):
        modules.append(nn.Linear(net_arch[idx], net_arch[idx + 1]))
        modules.append(activation_fn())

    if output_dim > 0:
        last_layer_dim = net_arch[-1] if len(net_arch) > 0 else input_dim
        modules.append(nn.Linear(last_layer_dim, output_dim))
    if squash_output:
        modules.append(nn.Tanh())
    return modules


class MlpExtractor(nn.Module):
    """
    Constructs an MLP that receives the output from a previous feature extractor (i.e. a CNN) or directly
    the observations (if no feature extractor is applied) as an input and outputs a latent representation
    for the policy and a value network.
    The ``net_arch`` parameter allows to specify the amount and size of the hidden layers and how many
    of them are shared between the policy network and the value network. It is assumed to be a list with the following
    structure:
    1. An arbitrary length (zero allowed) number of integers each specifying the number of units in a shared layer.
       If the number of ints is zero, there will be no shared layers.
    2. An optional dict, to specify the following non-shared layers for the value network and the policy network.
       It is formatted like ``dict(vf=[<value layer sizes>], pi=[<policy layer sizes>])``.
       If it is missing any of the keys (pi or vf), no non-shared layers (empty list) is assumed.
    For example to construct a network with one shared layer of size 55 followed by two non-shared layers for the value
    network of size 255 and a single non-shared layer of size 128 for the policy network, the following layers_spec
    would be used: ``[55, dict(vf=[255, 255], pi=[128])]``. A simple shared network topology with two layers of size 128
    would be specified as [128, 128].
    Adapted from Stable Baselines.
    :param feature_dim: Dimension of the feature vector (can be the output of a CNN)
    :param net_arch: The specification of the policy and value networks.
        See above for details on its formatting.
    :param activation_fn: The activation function to use for the networks.
    :param device:
    """

    def __init__(
        self,
        feature_dim: int,
        net_arch: List[Union[int, Dict[str, List[int]]]],
        activation_fn: Type[nn.Module],
        device: Union[th.device, str] = "auto",
    ):
        super().__init__()
        device = get_device(device)
        shared_net, policy_net, value_net = [], [], []
        policy_only_layers = [
        ]  # Layer sizes of the network that only belongs to the policy network
        value_only_layers = [
        ]  # Layer sizes of the network that only belongs to the value network
        last_layer_dim_shared = feature_dim

        # Iterate through the shared layers and build the shared parts of the network
        for layer in net_arch:
            if isinstance(layer, int):  # Check that this is a shared layer
                # TODO: give layer a meaningful name
                shared_net.append(nn.Linear(last_layer_dim_shared,
                                            layer))  # add linear of size layer
                shared_net.append(activation_fn())
                last_layer_dim_shared = layer
            else:
                assert isinstance(
                    layer, dict
                ), "Error: the net_arch list can only contain ints and dicts"
                if "pi" in layer:
                    assert isinstance(
                        layer["pi"], list
                    ), "Error: net_arch[-1]['pi'] must contain a list of integers."
                    policy_only_layers = layer["pi"]

                if "vf" in layer:
                    assert isinstance(
                        layer["vf"], list
                    ), "Error: net_arch[-1]['vf'] must contain a list of integers."
                    value_only_layers = layer["vf"]
                break  # From here on the network splits up in policy and value network

        last_layer_dim_pi = last_layer_dim_shared
        last_layer_dim_vf = last_layer_dim_shared

        # Build the non-shared part of the network
        for pi_layer_size, vf_layer_size in zip_longest(
                policy_only_layers, value_only_layers):
            if pi_layer_size is not None:
                assert isinstance(
                    pi_layer_size, int
                ), "Error: net_arch[-1]['pi'] must only contain integers."
                policy_net.append(nn.Linear(last_layer_dim_pi, pi_layer_size))
                policy_net.append(activation_fn())
                last_layer_dim_pi = pi_layer_size

            if vf_layer_size is not None:
                assert isinstance(
                    vf_layer_size, int
                ), "Error: net_arch[-1]['vf'] must only contain integers."
                value_net.append(nn.Linear(last_layer_dim_vf, vf_layer_size))
                value_net.append(activation_fn())
                last_layer_dim_vf = vf_layer_size

        # Save dim, used to create the distributions
        self.latent_dim_pi = last_layer_dim_pi
        self.latent_dim_vf = last_layer_dim_vf

        # Create networks
        # If the list of layers is empty, the network will just act as an Identity module
        self.shared_net = nn.Sequential(*shared_net).to(device)
        self.policy_net = nn.Sequential(*policy_net).to(device)
        self.value_net = nn.Sequential(*value_net).to(device)

    def forward(self, features: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        """
        :return: latent_policy, latent_value of the specified network.
            If all layers are shared, then ``latent_policy == latent_value``
        """
        shared_latent = self.shared_net(features)
        return self.policy_net(shared_latent), self.value_net(shared_latent)

    def forward_actor(self, features: th.Tensor) -> th.Tensor:
        return self.policy_net(self.shared_net(features))

    def forward_critic(self, features: th.Tensor) -> th.Tensor:
        return self.value_net(self.shared_net(features))


#=====================tactile===========================


class ResidualBlock(nn.Module):

    def __init__(self, in_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels,
                               out_channels=in_channels,
                               kernel_size=3,
                               stride=1,
                               padding=1)
        self.conv2 = nn.Conv2d(in_channels=in_channels,
                               out_channels=in_channels,
                               kernel_size=3,
                               stride=1,
                               padding=1)

    def forward(self, x):
        out = nn.ReLU()(x)
        out = self.conv1(out)
        out = nn.ReLU()(out)
        out = self.conv2(out)
        return out + x


class ImpalaBlock(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(ImpalaBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_channels,
                              out_channels=out_channels,
                              kernel_size=3,
                              stride=1,
                              padding=1)
        self.res1 = ResidualBlock(out_channels)
        self.res2 = ResidualBlock(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)(x)
        x = self.res1(x)
        x = self.res2(x)
        return x


class ImpalaModel(nn.Module):

    def __init__(self, observation_space, features_dim: int = 256, **kwargs):
        super(ImpalaModel, self).__init__()

        assert is_image_space(observation_space, check_channels=False), (
            "You should use NatureCNN "
            f"only with images not with {observation_space}\n"
            "(you are probably using `CnnPolicy` instead of `MlpPolicy` or `MultiInputPolicy`)\n"
            "If you are using a custom environment,\n"
            "please check it using our env checker:\n"
            "https://stable-baselines3.readthedocs.io/en/master/common/env_checker.html"
        )
        n_input_channels = observation_space.shape[0]
        self.block1 = ImpalaBlock(in_channels=n_input_channels,
                                  out_channels=16)
        self.block2 = ImpalaBlock(in_channels=16, out_channels=32)
        self.block3 = ImpalaBlock(in_channels=32, out_channels=32)
        self.fc = nn.Linear(in_features=32 * 8 * 8, out_features=256)

        self.output_dim = features_dim
        self.apply(xavier_uniform_init)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = nn.ReLU()(x)
        x = Flatten()(x)
        x = self.fc(x)
        x = nn.ReLU()(x)
        return x


def xavier_uniform_init(module, gain=1.0):
    if isinstance(module, nn.Linear) or isinstance(module, nn.Conv2d):
        nn.init.xavier_uniform_(module.weight.data, gain)
        nn.init.constant_(module.bias.data, 0)
    return module


class Flatten(nn.Module):

    def forward(self, x):
        return x.view(x.size(0), -1)


class CombinedExtractor(BaseFeaturesExtractor):
    """
    Combined feature extractor for Dict observation spaces.
    Builds a feature extractor for each key of the space. Input from each space
    is fed through a separate submodule (CNN or MLP, depending on input shape),
    the output features are concatenated and fed through additional MLP network ("combined").
    :param observation_space:
    :param cnn_output_dim: Number of features to output from each CNN submodule(s). Defaults to
        256 to avoid exploding network sizes.
    """

    def __init__(self,
                 observation_space: gym.spaces.Dict,
                 key: str,
                 features_extractor_class: Type[
                     BaseFeaturesExtractor] = FlattenExtractor,
                 cnn_output_dim: int = 256,
                 state_key="state",
                 state_mlp_size=(64, 64),
                 state_mlp_activation_fn=nn.ReLU,
                 augmentation=False,
                 cnn_type: int = 0,
                 dagger_tactile=False):
        # TODO we do not know features-dim here before going over all the items, so put something there. This is dirty!
        super().__init__(observation_space, features_dim=1)

        extractors = {}

        total_concat_size = 0
        self.augmentation = augmentation

        for key, subspace in observation_space.spaces.items():

            if is_image_space(subspace):

                extractors[key] = features_extractor_class(
                    subspace, features_dim=cnn_output_dim, cnn_type=cnn_type)
                total_concat_size += cnn_output_dim

            elif key == "relocate-point_cloud":
                pc_obs_dict = {}
                pc_obs_dict[key] = subspace
                extractors[key] = PointNetExtractor(
                    pc_obs_dict,
                    pc_key="relocate-point_cloud",
                    local_channels=(64, 128, 256),
                    global_channels=(256, ),
                    use_bn=True,
                )

                total_concat_size += extractors[key].features_dim

            elif key == state_key:

                # # The observation key is a vector, flatten it if needed
                # extractors[key] = nn.Flatten()
                # total_concat_size += get_flattened_obs_dim(subspace)

                self.state_space = observation_space[key]
                self.state_dim = self.state_space.shape[0]
                if dagger_tactile:
                    self.state_dim -= 1

                if len(state_mlp_size) == 0:
                    raise RuntimeError(f"State mlp size is empty")
                elif len(state_mlp_size) == 1:
                    net_arch = []
                else:
                    net_arch = state_mlp_size[:-1]
                output_dim = state_mlp_size[-1]

                output_dim = state_mlp_size[-1]

                total_concat_size += output_dim

                self.state_mlp = nn.Sequential(
                    *create_mlp(self.state_dim, output_dim, net_arch,
                                state_mlp_activation_fn))

                extractors[key] = self.state_mlp

        self.extractors = nn.ModuleDict(extractors)

        # Update the features dim manually
        self._features_dim = total_concat_size

        if self.augmentation:
            self.aug = RandomShiftsAug(4)

    def forward(self, observations: TensorDict) -> th.Tensor:
        encoded_tensor_list = []

        for key, extractor in self.extractors.items():

            if key == "relocate-point_cloud":

                pc_obs_dict = {}
                pc_obs_dict[key] = observations[key]
                encoded_tensor_list.append(extractor(pc_obs_dict))

            else:

                if key == "tactile_force":
                    continue

                if key == "tactile_image":

                    if self.augmentation:
                        encoded_tensor_list.append(
                            extractor(self.aug(observations[key])))

                    else:
                        encoded_tensor_list.append(extractor(
                            observations[key]))
                else:

                    encoded_tensor_list.append(
                        extractor(observations[key][:, :self.state_dim]))

        return th.cat(encoded_tensor_list, dim=1)


class RandomShiftsAug(nn.Module):

    def __init__(self, pad):
        super().__init__()
        self.pad = pad

    def forward(self, x):
        n, c, h, w = x.size()

        assert h == w
        padding = tuple([self.pad] * 4)
        x = F.pad(x, padding, 'replicate')
        eps = 1.0 / (h + 2 * self.pad)
        arange = torch.linspace(-1.0 + eps,
                                1.0 - eps,
                                h + 2 * self.pad,
                                device=x.device,
                                dtype=x.dtype)[:h]
        arange = arange.unsqueeze(0).repeat(h, 1).unsqueeze(2)
        base_grid = torch.cat([arange, arange.transpose(1, 0)], dim=2)
        base_grid = base_grid.unsqueeze(0).repeat(n, 1, 1, 1)

        shift = torch.randint(0,
                              2 * self.pad + 1,
                              size=(n, 1, 1, 2),
                              device=x.device,
                              dtype=x.dtype)
        shift *= 2.0 / (h + 2 * self.pad)

        grid = base_grid + shift

        return F.grid_sample(x,
                             grid,
                             padding_mode='zeros',
                             align_corners=False)


#=====================tactile===========================


class CombinedTactileGateExtractor(BaseFeaturesExtractor):
    """
    Combined feature extractor for Dict observation spaces.
    Builds a feature extractor for each key of the space. Input from each space
    is fed through a separate submodule (CNN or MLP, depending on input shape),
    the output features are concatenated and fed through additional MLP network ("combined").
    :param observation_space:
    :param cnn_output_dim: Number of features to output from each CNN submodule(s). Defaults to
        256 to avoid exploding network sizes.
    """

    def __init__(self,
                 observation_space: gym.spaces.Dict,
                 key: str,
                 features_extractor_class: Type[
                     BaseFeaturesExtractor] = FlattenExtractor,
                 cnn_output_dim: int = 256,
                 state_key="state",
                 state_mlp_size=(64, 64),
                 state_mlp_activation_fn=nn.ReLU,
                 augmentation=False,
                 cnn_type: int = 0):
        # TODO we do not know features-dim here before going over all the items, so put something there. This is dirty!
        super().__init__(observation_space, features_dim=1)

        extractors = {}

        total_concat_size = 0

        self.augmentation = augmentation

        for key, subspace in observation_space.spaces.items():

            if is_image_space(subspace):

                extractors[key] = features_extractor_class(
                    subspace, features_dim=cnn_output_dim, cnn_type=cnn_type)
                # total_concat_size += cnn_output_dim

            elif key == state_key:

                # # The observation key is a vector, flatten it if needed
                # extractors[key] = nn.Flatten()
                # total_concat_size += get_flattened_obs_dim(subspace)

                self.state_space = observation_space[key]
                self.state_dim = self.state_space.shape[0]

                if len(state_mlp_size) == 0:
                    raise RuntimeError(f"State mlp size is empty")
                elif len(state_mlp_size) == 1:
                    net_arch = []
                else:
                    net_arch = state_mlp_size[:-1]
                output_dim = state_mlp_size[-1]

                output_dim = state_mlp_size[-1]

                total_concat_size += output_dim

                self.state_mlp = nn.Sequential(
                    *create_mlp(self.state_dim, output_dim, net_arch,
                                state_mlp_activation_fn))

                extractors[key] = self.state_mlp

        self.extractors = nn.ModuleDict(extractors)
        self.TactileState_normalized = nn.Linear(
            cnn_output_dim + total_concat_size, total_concat_size)

        # Update the features dim manually
        self._features_dim = total_concat_size
        self.aug = RandomShiftsAug(4)

    def forward(self, observations: TensorDict) -> th.Tensor:

        tactile_index = torch.unique(
            torch.where(observations["tactile_force"] > 0.1)[0])
        state_index = torch.unique(
            torch.where(observations["tactile_force"] <= 0.1)[0])
        tactile_state_features = None
        state_features = None

        encoder = torch.zeros(
            (observations["tactile_image"].shape[0], self._features_dim),
            device="cuda")

        if len(tactile_index) != 0:

            if self.augmentation:
                tactile_features = self.extractors["tactile_image"](self.aug(
                    observations["tactile_image"][tactile_index]))

            else:

                tactile_features = self.extractors["tactile_image"](
                    observations["tactile_image"][tactile_index])

            state_features = self.extractors["oracle_state"](
                observations["oracle_state"][tactile_index])
            encoder[tactile_index] = self.TactileState_normalized(
                th.cat([tactile_features, state_features],
                       dim=1))  # normalized as same output
        if len(state_index) != 0:
            encoder[state_index] = self.extractors["oracle_state"](
                observations["oracle_state"][state_index])

        return encoder


class CombinedEnconderExtractor(BaseFeaturesExtractor):

    def __init__(self,
                 observation_space: gym.spaces.Dict,
                 key: str,
                 image_dimension: list,
                 features_extractor_class: Type[
                     BaseFeaturesExtractor] = FlattenExtractor,
                 encoder_type="pixel",
                 encoder_feature_dim: int = 256,
                 num_layers: int = 4,
                 num_filters: int = 32,
                 state_key="state",
                 state_mlp_size=(64, 64),
                 state_mlp_activation_fn=nn.ReLU):
        # TODO we do not know features-dim here before going over all the items, so put something there. This is dirty!
        super().__init__(observation_space, features_dim=1)

        extractors = {}

        total_concat_size = 0

        for key, subspace in observation_space.spaces.items():

            if is_image_space(subspace):

                self.encoder = make_encoder(encoder_type, subspace,
                                            encoder_feature_dim, num_layers,
                                            num_filters)

                extractors[key] = self.encoder
                total_concat_size += self.encoder.feature_dim
            elif key == "relocate-point_cloud":
                pc_obs_dict = {}
                pc_obs_dict[key] = subspace
                extractors[key] = PointNetExtractor(
                    pc_obs_dict,
                    pc_key="relocate-point_cloud",
                    local_channels=(64, 128, 256),
                    global_channels=(256, ),
                    use_bn=True,
                )

                total_concat_size += extractors[key].features_dim

            else:

                if key == state_key:

                    self.state_space = observation_space[key]
                    self.state_dim = self.state_space.shape[0]

                    if len(state_mlp_size) == 0:
                        raise RuntimeError(f"State mlp size is empty")
                    elif len(state_mlp_size) == 1:
                        net_arch = []
                    else:
                        net_arch = state_mlp_size[:-1]
                    output_dim = state_mlp_size[-1]

                    # # The observation key is a vector, flatten it if needed
                    # extractors[key] = nn.Flatten()
                    # total_concat_size += get_flattened_obs_dim(subspace)
                    output_dim = state_mlp_size[-1]

                    total_concat_size += output_dim

                    self.state_mlp = nn.Sequential(
                        *create_mlp(self.state_dim, output_dim, net_arch,
                                    state_mlp_activation_fn))

                    extractors[key] = self.state_mlp

        self.extractors = nn.ModuleDict(extractors)

        # Update the features dim manually
        self._features_dim = total_concat_size

    def forward(self, observations: TensorDict) -> th.Tensor:
        encoded_tensor_list = []

        for key, extractor in self.extractors.items():

            if key == "relocate-point_cloud":

                pc_obs_dict = {}
                pc_obs_dict[key] = observations[key]
                encoded_tensor_list.append(extractor(pc_obs_dict))

            else:

                encoded_tensor_list.append(extractor(observations[key]))
        return th.cat(encoded_tensor_list, dim=1)


#=====================tactile===========================

#=====================tactile===========================


class CombinedCURLExtractor(BaseFeaturesExtractor):

    def __init__(self,
                 observation_space: gym.spaces.Dict,
                 key: str,
                 image_dimension: list,
                 features_extractor_class: Type[
                     BaseFeaturesExtractor] = FlattenExtractor,
                 encoder_type="pixel",
                 encoder_feature_dim: int = 50,
                 num_layers: int = 4,
                 num_filters: int = 32,
                 state_key="state",
                 state_mlp_size=(64, 64),
                 state_mlp_activation_fn=nn.ReLU):
        # TODO we do not know features-dim here before going over all the items, so put something there. This is dirty!
        super().__init__(observation_space, features_dim=1)

        extractors = {}

        total_concat_size = 0

        for key, subspace in observation_space.spaces.items():

            if is_image_space(subspace):

                self.encoder = make_encoder(encoder_type, subspace,
                                            encoder_feature_dim, num_layers,
                                            num_filters)

                extractors[key] = self.encoder
                total_concat_size += self.encoder.feature_dim
            elif key == "relocate-point_cloud":
                pc_obs_dict = {}
                pc_obs_dict[key] = subspace
                extractors[key] = PointNetExtractor(
                    pc_obs_dict,
                    pc_key="relocate-point_cloud",
                    local_channels=(64, 128, 256),
                    global_channels=(256, ),
                    use_bn=True,
                )

                total_concat_size += extractors[key].features_dim

            else:

                if key == state_key:

                    self.state_space = observation_space[key]
                    self.state_dim = self.state_space.shape[0]

                    if len(state_mlp_size) == 0:
                        raise RuntimeError(f"State mlp size is empty")
                    elif len(state_mlp_size) == 1:
                        net_arch = []
                    else:
                        net_arch = state_mlp_size[:-1]
                    output_dim = state_mlp_size[-1]

                    # # The observation key is a vector, flatten it if needed
                    # extractors[key] = nn.Flatten()
                    # total_concat_size += get_flattened_obs_dim(subspace)
                    output_dim = state_mlp_size[-1]

                    total_concat_size += output_dim

                    self.state_mlp = nn.Sequential(
                        *create_mlp(self.state_dim, output_dim, net_arch,
                                    state_mlp_activation_fn))

                    extractors[key] = self.state_mlp

        self.extractors = nn.ModuleDict(extractors)

        # Update the features dim manually
        self._features_dim = total_concat_size

    def forward(self, observations: TensorDict) -> th.Tensor:
        encoded_tensor_list = []

        for key, extractor in self.extractors.items():

            if key == "relocate-point_cloud":

                pc_obs_dict = {}
                pc_obs_dict[key] = observations[key]
                encoded_tensor_list.append(extractor(pc_obs_dict))

            else:

                encoded_tensor_list.append(extractor(observations[key]))
        return th.cat(encoded_tensor_list, dim=1)


#=====================tactile===========================


def get_actor_critic_arch(
    net_arch: Union[List[int], Dict[str, List[int]]]
) -> Tuple[List[int], List[int]]:
    """
    Get the actor and critic network architectures for off-policy actor-critic algorithms (SAC, TD3, DDPG).
    The ``net_arch`` parameter allows to specify the amount and size of the hidden layers,
    which can be different for the actor and the critic.
    It is assumed to be a list of ints or a dict.
    1. If it is a list, actor and critic networks will have the same architecture.
        The architecture is represented by a list of integers (of arbitrary length (zero allowed))
        each specifying the number of units per layer.
       If the number of ints is zero, the network will be linear.
    2. If it is a dict,  it should have the following structure:
       ``dict(qf=[<critic network architecture>], pi=[<actor network architecture>])``.
       where the network architecture is a list as described in 1.
    For example, to have actor and critic that share the same network architecture,
    you only need to specify ``net_arch=[256, 256]`` (here, two hidden layers of 256 units each).
    If you want a different architecture for the actor and the critic,
    then you can specify ``net_arch=dict(qf=[400, 300], pi=[64, 64])``.
    .. note::
        Compared to their on-policy counterparts, no shared layers (other than the features extractor)
        between the actor and the critic are allowed (to prevent issues with target networks).
    :param net_arch: The specification of the actor and critic networks.
        See above for details on its formatting.
    :return: The network architectures for the actor and the critic
    """
    if isinstance(net_arch, list):
        actor_arch, critic_arch = net_arch, net_arch
    else:
        assert isinstance(
            net_arch, dict
        ), "Error: the net_arch can only contain be a list of ints or a dict"
        assert "pi" in net_arch, "Error: no key 'pi' was provided in net_arch for the actor network"
        assert "qf" in net_arch, "Error: no key 'qf' was provided in net_arch for the critic network"
        actor_arch, critic_arch = net_arch["pi"], net_arch["qf"]
    return actor_arch, critic_arch


class PointNetExtractor(BaseFeaturesExtractor):
    """
    :param observation_space:
    """

    def __init__(self,
                 observation_space: gym.spaces.Dict,
                 pc_key: str,
                 feat_key: Optional[str] = None,
                 use_bn=True,
                 local_channels=(64, 128, 256),
                 global_channels=(256, ),
                 one_hot_dim=0):
        if feat_key is not None:
            if feat_key not in list(observation_space.keys()):
                raise RuntimeError(
                    f"Feature key {feat_key} not in observation space.")
        if pc_key not in list(observation_space.keys()):
            raise RuntimeError(
                f"Point cloud key {pc_key} not in observation space.")

        # Point cloud input should have size (n, 3), spec size (n, 3), feat size (n, m)
        self.pc_key = pc_key
        self.has_feat = feat_key is not None
        self.feat_key = feat_key
        pc_spec = observation_space[pc_key]
        pc_dim = pc_spec.shape[1]
        if self.has_feat:
            feat_spec = observation_space[feat_key]
            feat_dim = feat_spec.shape[1]
        else:
            feat_dim = 0
        features_dim = global_channels[-1]

        super().__init__(observation_space, features_dim)

        n_input_channels = pc_dim + feat_dim + one_hot_dim
        self.point_net = PointNet(n_input_channels,
                                  local_channels=local_channels,
                                  global_channels=global_channels,
                                  use_bn=use_bn)
        self.n_input_channels = n_input_channels
        self.n_output_channels = self.point_net.out_channels

    def forward(self, observations: TensorDict) -> th.Tensor:
        points = torch.transpose(observations[self.pc_key], 1, 2)
        if self.has_feat:
            feats = torch.transpose(observations[self.feat_key], 1, 2)
        else:
            feats = None
        return self.point_net(points, feats)["feature"]


class PointNetImaginationExtractor(PointNetExtractor):

    def __init__(self,
                 observation_space: gym.spaces.Dict,
                 pc_key: str,
                 use_bn=True,
                 local_channels=(64, 128, 256),
                 global_channels=(256, ),
                 imagination_keys=("imagination_robot", ),
                 state_key="state",
                 state_mlp_size=(64, 64),
                 state_mlp_activation_fn=nn.ReLU):
        self.imagination_key = imagination_keys
        self.imagination_one_hot = []
        if len(imagination_keys) >= 1:
            one_hot_dim = 4
        else:
            one_hot_dim = 0  # Vanilla PointNet

        # Init state representation
        self.use_state = state_key is not None
        self.state_key = state_key
        if self.use_state:
            if state_key not in observation_space.spaces.keys():
                raise RuntimeError(
                    f"State key {state_key} not in observation space: {observation_space}"
                )
            self.state_space = observation_space[self.state_key]

        super().__init__(observation_space,
                         pc_key,
                         None,
                         use_bn=use_bn,
                         local_channels=local_channels,
                         global_channels=global_channels,
                         one_hot_dim=one_hot_dim)

        # One hot vector for imagination
        num_class = 4
        device = next(self.parameters()).device
        for key in imagination_keys:
            if key not in list(observation_space.keys()):
                raise RuntimeError(
                    f"Imagination key {key} not in observation space.")
            if key == "imagination_robot":
                img_type = 1
            elif key == "imagination_goal":
                img_type = 2
            else:
                raise NotImplementedError
            num_points = observation_space[key].shape[0]
            tensor_img_type = torch.ones(
                num_points, dtype=torch.long, device=device) * img_type
            self.imagination_one_hot.append(
                F.one_hot(tensor_img_type, num_class).to(torch.float32))

        # One hot vector for observation
        num_points = observation_space[pc_key].shape[0]
        obs_tensor_img_type = torch.zeros(num_points,
                                          num_class,
                                          dtype=torch.float32,
                                          device=device)
        self.obs_pc_one_hot = obs_tensor_img_type

        self.img_feats = torch.transpose(
            torch.cat([self.obs_pc_one_hot] + self.imagination_one_hot), 0,
            1).unsqueeze(0)  # (1, 4, p)

        # State MLP
        if self.use_state:
            self.state_dim = self.state_space.shape[0]
            if len(state_mlp_size) == 0:
                raise RuntimeError(f"State mlp size is empty")
            elif len(state_mlp_size) == 1:
                net_arch = []
            else:
                net_arch = state_mlp_size[:-1]
            output_dim = state_mlp_size[-1]

            self.n_output_channels = self.point_net.out_channels + output_dim
            self._features_dim = self.n_output_channels
            self.state_mlp = nn.Sequential(*create_mlp(
                self.state_dim, output_dim, net_arch, state_mlp_activation_fn))

    def forward(self, observations: TensorDict) -> th.Tensor:
        points = torch.transpose(observations[self.pc_key], 1, 2)
        batch_size = points.shape[0]
        if self.img_feats.device != points.device:
            self.img_feats = self.img_feats.to(points.device)
        if len(self.imagination_key) > 0:
            img_points = []
            for key in self.imagination_key:
                img_points.append(observations[key].transpose(1, 2))
            points = torch.cat([points] + img_points, dim=2)
            feats = torch.tile(self.img_feats, (batch_size, 1, 1))
        else:
            feats = None

        pn_feat = self.point_net(points, feats)["feature"]
        if self.use_state:
            state_feat = self.state_mlp(observations[self.state_key])
            return torch.cat([pn_feat, state_feat], dim=-1)
        else:
            return pn_feat


class PointNetStateExtractor(PointNetExtractor):

    def __init__(self,
                 observation_space: gym.spaces.Dict,
                 pc_key: str,
                 use_bn=True,
                 local_channels=(64, 128, 256),
                 global_channels=(256, ),
                 state_key="state",
                 state_mlp_size=(64, 64),
                 state_mlp_activation_fn=nn.ReLU):
        self.state_key = state_key
        if state_key not in observation_space.spaces.keys():
            raise RuntimeError(
                f"State key {state_key} not in observation space: {observation_space}"
            )
        self.state_space = observation_space[self.state_key]

        super().__init__(observation_space,
                         pc_key,
                         None,
                         use_bn=use_bn,
                         local_channels=local_channels,
                         global_channels=global_channels,
                         one_hot_dim=0)

        self.state_dim = self.state_space.shape[0]
        if len(state_mlp_size) == 0:
            raise RuntimeError(f"State mlp size is empty")
        elif len(state_mlp_size) == 1:
            net_arch = []
        else:
            net_arch = state_mlp_size[:-1]
        output_dim = state_mlp_size[-1]

        self.n_output_channels = self.point_net.out_channels + output_dim
        self._features_dim = self.n_output_channels
        self.state_mlp = nn.Sequential(*create_mlp(
            self.state_dim, output_dim, net_arch, state_mlp_activation_fn))

    def forward(self, observations: TensorDict) -> th.Tensor:
        points = torch.transpose(observations[self.pc_key], 1, 2)
        if self.has_feat:
            feats = torch.transpose(observations[self.feat_key], 1, 2)
        else:
            feats = None

        pn_feat = self.point_net(points, feats)["feature"]
        state_feat = self.state_mlp(observations[self.state_key])
        return torch.cat([pn_feat, state_feat], dim=-1)


class PixelDelta2DEncoder(nn.Module):
    """Flare encoder of pixels observations.
        ref:https://github.com/WendyShang/flare/blob/ef12a7f4b2fc639d491ca63648ea1e8c58d07263/encoder.py#L276
    """

    def __init__(self,
                 observation_space: gym.spaces.Box,
                 features_dim: int = 512,
                 channels=[16, 32, 32],
                 num_layers=2,
                 num_filters=32,
                 output_logits=False,
                 image_channel=12,
                 stack=False,
                 flow=False):
        super().__init__()

        # assert is_image_space(observation_space, check_channels=False), (
        #     "You should use PixelDelta2DEncoder "
        #     f"only with images not with {observation_space}\n"
        #     "(you are probably using `CnnPolicy` instead of `MlpPolicy` or `MultiInputPolicy`)\n"
        #     "If you are using a custom environment,\n"
        #     "please check it using our env checker:\n"
        #     "https://stable-baselines3.readthedocs.io/en/master/common/env_checker.html"
        # )

        observation_space = observation_space.shape
        self.obs_shape = observation_space
        self.feature_dim = features_dim
        self.num_layers = num_layers
        self.image_channel = image_channel
        self.stack = stack
        self.flow = flow

        assert not (self.stack and self.flow)

        time_step = observation_space[0]

        self.convs = nn.ModuleList(
            [nn.Conv2d(self.image_channel, num_filters, 3, stride=2)])
        self.convs.append(nn.Conv2d(num_filters, num_filters, 3, stride=1))
        self.convs.append(nn.Conv2d(num_filters, num_filters, 3, stride=1))
        for i in range(2, num_layers - 1):
            self.convs.append(nn.Conv2d(num_filters, num_filters, 3, stride=1))
        self.outputs = dict()

        x = torch.randn([10] + [num_layers] + [self.image_channel] +
                        [observation_space[-2]] + [observation_space[-1]])
        #
        self.out_dim = self.forward_conv(x, flatten=True).shape[-1]

        print('conv output dim: ' + str(self.out_dim))

        self.fc = nn.Linear(self.out_dim, self.feature_dim)
        self.ln = nn.LayerNorm(self.feature_dim)

        self.output_logits = output_logits

    def reparameterize(self, mu, logstd):
        std = torch.exp(logstd)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward_conv(self, obs, flatten=True):

        if abs(obs.max()) > 1.:
            obs = obs / 255.

        time_step = obs.shape[1]
        # obs = obs.reshape(obs.shape[0], time_step, self.image_channel,
        #                obs.shape[-2], obs.shape[-1])
        obs = obs.contiguous().view(obs.shape[0] * time_step,
                                    self.image_channel, obs.shape[-2],
                                    obs.shape[-1])

        self.outputs['obs'] = obs

        conv = torch.relu(self.convs[0](obs))
        self.outputs['conv1'] = conv

        conv = torch.relu(self.convs[1](conv))
        self.outputs['conv%s' % (1 + 1)] = conv

        for i in range(2, self.num_layers):
            conv = torch.relu(self.convs[i](conv))
            self.outputs['conv%s' % (i + 1)] = conv

        conv = conv.view(
            conv.size(0) // time_step, time_step, conv.size(1), conv.size(2),
            conv.size(3))

        if self.stack:

            conv = conv.view(conv.size(0),
                             conv.size(1) * conv.size(2), conv.size(3),
                             conv.size(4))
        elif self.flow:  # diff

            conv_current = conv[:, 1:, :, :, :]
            conv_prev = conv_current - conv[:, :time_step -
                                            1, :, :, :].detach()

            conv = conv_prev.view(conv_prev.size(0),
                                  conv_prev.size(1) * conv_prev.size(2),
                                  conv_prev.size(3), conv_prev.size(4))

        else:
            conv_current = conv[:, 1:, :, :, :]
            conv_prev = conv_current - conv[:, :time_step -
                                            1, :, :, :].detach()
            conv = torch.cat([conv_current, conv_prev], axis=1)
            conv = conv.view(conv.size(0),
                             conv.size(1) * conv.size(2), conv.size(3),
                             conv.size(4))

        if not flatten:
            return conv
        else:
            conv = conv.view(conv.size(0), -1)
            return conv

    def forward(self, obs, detach=False):

        h = self.forward_conv(obs)

        if detach:
            h = h.detach()

        try:
            h_fc = self.fc(h)
        except:
            print(obs.shape)
            print(h.shape)
            assert False
        self.outputs['fc'] = h_fc

        h_norm = self.ln(h_fc)
        self.outputs['ln'] = h_norm

        if self.output_logits:
            out = h_norm
        else:
            out = torch.tanh(h_norm)
            self.outputs['tanh'] = out

        return out


class CombinedExtractorPixelDelta(BaseFeaturesExtractor):
    """
    Combined feature extractor for Dict observation spaces.
    Builds a feature extractor for each key of the space. Input from each space
    is fed through a separate submodule (CNN or MLP, depending on input shape),
    the output features are concatenated and fed through additional MLP network ("combined").
    :param observation_space:
    :param cnn_output_dim: Number of features to output from each CNN submodule(s). Defaults to
        256 to avoid exploding network sizes.
    """

    def __init__(self,
                 observation_space: gym.spaces.Dict,
                 key: str,
                 features_extractor_class: Type[
                     BaseFeaturesExtractor] = FlattenExtractor,
                 features_dim: int = 256,
                 state_key="state",
                 state_mlp_size=(64, 64),
                 state_mlp_activation_fn=nn.ReLU,
                 augmentation=False,
                 channels=[16, 32, 32],
                 num_layers: int = 1,
                 num_filters: int = 32,
                 output_logits: bool = False,
                 image_channel: int = 12,
                 stack: bool = False,
                 flow: bool = False):
        # TODO we do not know features-dim here before going over all the items, so put something there. This is dirty!
        super().__init__(observation_space, features_dim=1)

        extractors = {}

        total_concat_size = 0
        self.augmentation = augmentation

        for key, subspace in observation_space.spaces.items():

            if len(subspace.shape) >= 3:

                extractors[key] = features_extractor_class(
                    subspace,
                    features_dim,
                    channels=channels,
                    num_layers=num_layers,
                    num_filters=num_filters,
                    output_logits=output_logits,
                    image_channel=image_channel,
                    stack=stack,
                    flow=flow)
                total_concat_size += features_dim

            elif key == "relocate-point_cloud":
                pc_obs_dict = {}
                pc_obs_dict[key] = subspace
                extractors[key] = PointNetExtractor(
                    pc_obs_dict,
                    pc_key="relocate-point_cloud",
                    local_channels=(64, 128, 256),
                    global_channels=(256, ),
                    use_bn=True,
                )

                total_concat_size += extractors[key].features_dim

            elif key == state_key:

                # # The observation key is a vector, flatten it if needed
                # extractors[key] = nn.Flatten()
                # total_concat_size += get_flattened_obs_dim(subspace)

                self.state_space = observation_space[key]
                self.state_dim = self.state_space.shape[0]

                if len(state_mlp_size) == 0:
                    raise RuntimeError(f"State mlp size is empty")
                elif len(state_mlp_size) == 1:
                    net_arch = []
                else:
                    net_arch = state_mlp_size[:-1]
                output_dim = state_mlp_size[-1]

                output_dim = state_mlp_size[-1]

                total_concat_size += output_dim

                self.state_mlp = nn.Sequential(
                    *create_mlp(self.state_dim, output_dim, net_arch,
                                state_mlp_activation_fn))

                extractors[key] = self.state_mlp

        self.extractors = nn.ModuleDict(extractors)

        # Update the features dim manually
        self._features_dim = total_concat_size

        if self.augmentation:
            self.aug = RandomShiftsAug(4)

    def forward(self, observations: TensorDict) -> th.Tensor:

        encoded_tensor_list = []

        for key, extractor in self.extractors.items():

            if key == "relocate-point_cloud":

                pc_obs_dict = {}
                pc_obs_dict[key] = observations[key]
                encoded_tensor_list.append(extractor(pc_obs_dict))

            else:

                if key == "tactile_force":
                    continue

                if key == "tactile_image":

                    if self.augmentation:
                        encoded_tensor_list.append(
                            extractor(self.aug(observations[key])))

                    else:
                        encoded_tensor_list.append(extractor(
                            observations[key]))
                else:

                    encoded_tensor_list.append(extractor(observations[key]))

        return th.cat(encoded_tensor_list, dim=1)
