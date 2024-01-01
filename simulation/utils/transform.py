import os
import numpy as np

import torch
import torch.optim as optim
import torch.nn as nn

import pytorch3d.transforms as p3d_t
import pdb


def tf3d_between(pose1, pose2, device=None):
    """
    Relative transform of pose2 in pose1 frame, i.e. T12 = T1^{1}*T2
    :param pose1: n x 3 tensor [x,y,yaw]
    :param pose2: n x 3 tensor [x,y,yaw]
    :return: pose12 n x 3 tensor [x,y,yaw]
    """
    rot1 = pose1[:, :4]  #obj
    rot2 = pose2[:, :4]  #tip

    t1 = pose1[:, 4:]
    t2 = pose2[:, 4:]

    R1 = p3d_t.quaternion_to_matrix(rot1)
    R2 = p3d_t.quaternion_to_matrix(rot2)
    R1t = torch.inverse(R1)

    R12 = torch.matmul(R1t, R2)
    rot12 = p3d_t.matrix_to_euler_angles(R12, "XYZ")
    t12 = torch.matmul(R1t, (t2 - t1)[:, :, None])
    t12 = t12[:, :, 0]

    relative_pose = torch.cat((rot12, t12), axis=1)

    return relative_pose.numpy()