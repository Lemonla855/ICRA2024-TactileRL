import numpy as np
from sapien.core import Pose
from scipy.spatial.transform import Rotation as R
import pdb
from scipy.spatial.transform import Rotation as R

CAM2ROBOT = Pose.from_transformation_matrix(
    np.array([[0.60346958, 0.36270068, -0.7101216, 0.962396],
              [0.7960018, -0.22156729, 0.56328419, -0.35524235],
              [0.04696384, -0.90518294, -0.42241951, 0.31896536],
              [0., 0., 0., 1.]]))

r = R.from_quat(CAM2ROBOT.q[[1, 2, 3, 0]])
cam_angle = r.as_euler('xyz', degrees=True)
cam_postion = CAM2ROBOT.p

cam_postion[0] = 0.4
cam_postion[1] = -0.5
cam_postion[2] = 0.3
# pdb.set_trace()
cam_angle[2] = 0
cam_angle[1] = 0
cam_angle[0] = -90

r = R.from_euler('xyz', cam_angle, degrees=True)
cam_angle = r.as_quat()[[3, 0, 1, 2]]
CAM2ROBOT_EXTRINSIC = Pose(cam_postion, cam_angle)
# cam_angle[0] = -100
cam_postion[1] = -0.6
cam_postion[2] = 0.3
CAM2ROBOT_INSERTION = Pose(cam_postion, cam_angle)

# pivot point cloud camera
# CAM2ROBOT_PIVOT = Pose.from_transformation_matrix(
#     [[0.98847067, 0.00188947, 0.15140067, 0.18654601],
#      [-0.15085069, -0.07376564, 0.98580054, -1.00740585],
#      [0.0130308, -0.99727381, -0.07263014, 0.3138825], [0., 0., 0., 1.]])
CAM2ROBOT_PIVOT = Pose.from_transformation_matrix(
    [[0.9988839, -0.00263637, -0.0471594, 0.31261481],
     [0.04686784, -0.06859463, 0.99654312, -0.79978504],
     [-0.00586214, -0.99764113, -0.06839451, 0.27571799], [0., 0., 0., 1.]])

DESK2ROBOT_Z_AXIS = 0
# Relocate
RELOCATE_BOUND = [0.2, 0.8, -0.2, 0.2, DESK2ROBOT_Z_AXIS + 0.005, 0.6]

# TODO:
# ROBOT2BASE = Pose(p=np.array([-0.55, 0., -DESK2ROBOT_Z_AXIS]))
ROBOT2BASE = Pose(p=np.array([-0.40, 0., -DESK2ROBOT_Z_AXIS]))

# Table size
# TABLE_XY_SIZE = np.array([0.6, 1.2])
TABLE_XY_SIZE = np.array([4, 4])
# TABLE_ORIGIN = np.array([0, -0.15])
TABLE_ORIGIN = np.array([-0.0, -0.0])
