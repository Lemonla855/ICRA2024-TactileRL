import torch
import torch.nn as nn
import pdb
from torchvision.models import vgg19_bn, vgg16_bn, inception_v3, alexnet, resnet18, resnet34, resnet50
import torch as th
import gym


def tie_weights(src, trg):
    assert type(src) == type(trg)
    trg.weight = src.weight
    trg.bias = src.bias


OUT_DIM = {2: 29, 4: 25, 6: 21}


class PixelEncoder(nn.Module):
    """Convolutional encoder of pixels observations."""

    def __init__(self, obs_space, feature_dim, num_layers=2, num_filters=32):
        super().__init__()

        obs_shape = obs_space.shape

        assert len(obs_shape) == 3
        self.outputs = dict()

        self.feature_dim = feature_dim
        self.num_layers = num_layers

        self.convs = nn.ModuleList(
            [nn.Conv2d(obs_shape[0], num_filters, 3, stride=2)])
        for i in range(num_layers - 1):
            self.convs.append(nn.Conv2d(num_filters, num_filters, 3, stride=1))

        # with th.no_grad():
        #     self.n_flatten = self.forward_conv(
        #         th.as_tensor(obs_space.sample()[None]).float()).shape[1]
        out_dim = OUT_DIM[num_layers]
        self.fc = nn.Linear(num_filters * out_dim * out_dim, self.feature_dim)

        # self.fc = nn.Linear(self.n_flatten, self.feature_dim)
        self.ln = nn.LayerNorm(self.feature_dim)

    def reparameterize(self, mu, logstd):
        std = th.exp(logstd)
        eps = th.randn_like(std)
        return mu + eps * std

    def forward_conv(self, obs):
        obs = obs / 255.

        self.outputs['obs'] = obs

        conv = th.relu(self.convs[0](obs))
        self.outputs['conv1'] = conv

        for i in range(1, self.num_layers):
            conv = th.relu(self.convs[i](conv))
            self.outputs['conv%s' % (i + 1)] = conv

        h = conv.view(conv.size(0), -1)

        return h

    def forward(self, obs, detach=False):
        h = self.forward_conv(obs)

        if detach:
            h = h.detach()

        h_fc = self.fc(h)
        self.outputs['fc'] = h_fc

        h_norm = self.ln(h_fc)
        self.outputs['ln'] = h_norm

        out = th.tanh(h_norm)
        self.outputs['tanh'] = out

        return out

    def copy_conv_weights_from(self, source):
        """Tie convolutional layers"""
        # only tie conv layers
        for i in range(self.num_layers):
            tie_weights(src=source.convs[i], trg=self.convs[i])

    def log(self, L, step, log_freq):
        if step % log_freq != 0:
            return

        for k, v in self.outputs.items():
            L.log_histogram('train_encoder/%s_hist' % k, v, step)
            if len(v.shape) > 2:
                L.log_image('train_encoder/%s_img' % k, v[0], step)

        for i in range(self.num_layers):
            L.log_param('train_encoder/conv%s' % (i + 1), self.convs[i], step)
        L.log_param('train_encoder/fc', self.fc, step)
        L.log_param('train_encoder/ln', self.ln, step)


class IdentityEncoder(nn.Module):

    def __init__(self, obs_shape, feature_dim, num_layers, num_filters):
        super().__init__()

        assert len(obs_shape) == 1
        self.feature_dim = obs_shape[0]

    def forward(self, obs, detach=False):
        return obs

    def copy_conv_weights_from(self, source):
        pass

    def log(self, L, step, log_freq):
        pass


_AVAILABLE_ENCODERS = {'pixel': PixelEncoder, 'identity': IdentityEncoder}


def make_encoder(encoder_type, obs_shape, feature_dim, num_layers,
                 num_filters):
    assert encoder_type in _AVAILABLE_ENCODERS
    return _AVAILABLE_ENCODERS[encoder_type](obs_shape, feature_dim,
                                             num_layers, num_filters)


def Resnet18(output_dim: int = 256):
    features = resnet18(pretrained=True)
    features.fc = nn.Linear(features.fc.in_features, output_dim)

    return features
