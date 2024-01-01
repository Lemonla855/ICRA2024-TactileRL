import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import pdb
import numpy as np
import os
from torch.utils.data import Dataset, DataLoader
import gym
from pathlib import Path
from torchvision import transforms
from hand_env_utils.arg_utils import *
from stable_baselines3.common.utils import get_device
import shutil
from estimator.convnext import ConvNeXt


class AngleNet(nn.Module):

    def __init__(self, features_dim=256, observation_space=None):
        super(AngleNet, self).__init__()
        self.n_input_channels = int(observation_space.shape[0])

        observation_space = gym.spaces.Box(low=0,
                                           high=255,
                                           dtype=np.uint8,
                                           shape=(self.n_input_channels,
                                                  observation_space.shape[1],
                                                  observation_space.shape[2]))

        self.cnn = nn.Sequential(
            nn.Conv2d(self.n_input_channels,
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

        with torch.no_grad():
            n_flatten = self.cnn(
                torch.as_tensor(
                    observation_space.sample()[None]).float()).shape[1]

        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim),
                                    nn.ReLU())

        self.predictor = nn.Sequential(nn.Linear(features_dim, 1), nn.ReLU())

    def forward(self, observations: torch.Tensor) -> torch.Tensor:

        return self.predictor(self.linear(self.cnn(observations)))


class TorchDataset(Dataset):

    def __init__(self,
                 train_time,
                 filename,
                 transform=None,
                 num_file_sample=25):
        self.num_file_sample = num_file_sample

        len_angle = self.read_file(train_time, filename)

        self.train_time = train_time
        self.filename = filename

        self.name = filename + '-' + str(500)

        self.len = len_angle
        self.transform = transform

        self.device = get_device()

    def __getitem__(self, index):

        angle = np.clip(
            int(
                np.load("estimator/dataset/" + self.train_time + "/" +
                        self.name + "/angle/%d.npy" % index) / np.pi * 180 -
                90), 0, 119)

        label = np.zeros(120)
        label[angle] = 1

        img = np.load("estimator/dataset/" + self.train_time + "/" +
                      self.name +
                      "/tactile_image/%d.npy" % index)[np.random.choice(
                          [0, 1])][None]

        img = np.sign(img)

        img, label = torch.as_tensor(
            img, dtype=torch.float32), torch.as_tensor(label,
                                                       dtype=torch.float32)
        if self.transform is not None:
            img = self.transform(img)

        return img.to(self.device), label.to(self.device)

    def __len__(self):

        data_len = self.len
        return data_len

    def read_file(self, train_time, filename):
        # image_label_list = np.load(filename + "/anglenet_angle.npy")
        # image_list = np.load(filename + "/anglenet_tactile_image.npy")
        model_name = filename

        angle_list = []
        tactile_image_list = []

        seed_index = 0

        self.label_dir = [[] for i in range(5)]

        for seed in [100, 200, 300, 400, 500]:
            name = model_name + '-' + str(seed)
            if not os.path.exists("estimator/dataset/" + train_time + "/" +
                                  name):
                continue

            num_files = len(
                os.listdir("estimator/dataset/" + train_time + "/" + name +
                           "/angle"))

        #     for i in range(self.num_file_sample):

        #         tactile_image = np.load("estimator/dataset/" + train_time +
        #                                 "/" + name +
        #                                 "/tactile_image/%d.npy" % int(i * 2),
        #                                 allow_pickle=True)
        #         tactile_image = tactile_image.reshape(-1,
        #                                               tactile_image.shape[2],
        #                                               tactile_image.shape[3],
        #                                               tactile_image.shape[4])

        #         angle = np.load("estimator/dataset/" + train_time + "/" +
        #                         name + "/angle/%d.npy" % (i * 2),
        #                         allow_pickle=True)
        #         angle = angle.reshape(-1)

        #         self.per_batch = self.num_file_sample * 100 * 200

        #         self.label_dir[
        #             seed_index] = "estimator/dataset/" + train_time + "/" + name

        #         angle_list.append(angle)
        #         tactile_image_list.append(tactile_image)

        # angle_list = np.concatenate(angle_list, axis=0)
        # tactile_image = np.concatenate(tactile_image_list, axis=0)

        # seed_index += 1

        return 200000  #, tactile_image / 255


def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    # Set the model to training mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        # Compute prediction and loss

        pred = model(X)

        loss = loss_fn(pred, y)
        # loss = torch.mean(abs(y - pred)**2)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]", )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_time', type=str, default="all_kind"
                        )  #"ppo-4118-partnet_bottle_noise_random_lefthand-500"

    parser.add_argument(
        '--name',
        type=str,
        default=
        "ppo-any_train-2-all_kind_multi_noise_random_righthanddown-tactile_crop_mask"
    )  #"ppo-4118-partnet_bottle_noise_random_lefthand-500"

    args = parser.parse_args()

    learning_rate = 1e-4
    filename = args.name
    result_path = Path("estimator/dataset/" + args.train_time + "/" +
                       filename + "/model")
    result_path.mkdir(exist_ok=True, parents=True)

    transform = transforms.Compose([
        transforms.RandomAffine(
            degrees=0, translate=(0.0, 0.0),
            scale=(0.5, 1.0)),  # Apply random affine transformations
        # transforms.ToTensor(),  # Convert the image to a PyTorch tensor
        # transforms.Normalize(mean=[0.485, 0.456, 0.406],
        #                      std=[0.229, 0.224, 0.225])  # Normalize the image
        transforms.RandomErasing(scale=(0.03, 0.3)),
    ])

    dataset = TorchDataset(args.train_time, filename, transform=transform)

    # Shuffle the dataset
    shuffle_dataset = True
    random_seed = 42
    if shuffle_dataset:
        torch.manual_seed(random_seed)
        indices = torch.randperm(len(dataset))
        dataset = torch.utils.data.Subset(dataset, indices)

    # Create a data loader
    batch_size = 1024
    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=batch_size,
                                             shuffle=True)

    # Iterate over the dataset
    loss_fn = nn.CrossEntropyLoss()
    # model = AngleNet(observation_space=np.zeros((2, 64, 64))).to(get_device())

    model = ConvNeXt(in_chans=1, num_classes=120).to(get_device())
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    epochs = 100
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train_loop(dataloader, model, loss_fn, optimizer)
        if t % 5 == 0:
            torch.save(
                model.state_dict(), "estimator/dataset/" + args.train_time +
                "/" + filename + "/model/model_%d.pth" % t)

    print("Done!")
