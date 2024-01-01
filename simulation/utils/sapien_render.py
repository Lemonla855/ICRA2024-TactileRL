import numpy as np

import pdb
import matplotlib.pyplot as plt
import torch
from typing import Union
import trimesh
from omegaconf import OmegaConf
from sapien.utils import Viewer
# 3D transformations functions
from pytorch3d.transforms import quaternion_to_matrix, Transform3d, quaternion_invert, matrix_to_quaternion, quaternion_multiply
import cv2

conf = OmegaConf.load("/home/lme/Documents/sapien_task/conf/rendering.yaml")

height, width = 160, 120


#==============================================================
# update render result for GUI if needed
#==============================================================
def updateGUI(colors: np.ndarray, depths: np.ndarray) -> None:
    """
    Update images for visualization
    """

    # concatenate colors horizontally (axis=1)
    color = np.concatenate(colors, axis=1)

    cv2.imshow("color and depth", cv2.cvtColor(color, cv2.COLOR_RGB2BGR))
    cv2.waitKey(1)


#========================================================================================
# init lighting(refer from tacto)
#========================================================================================
def init_light() -> None:
    """
    init the light information
    """
    # transparancy
    alphas = torch.ones((1, 64, 64), ) * 1.0
    # light position
    # Load light from config file
    light = conf.sensor.lights

    origin = np.array(light.origin, dtype=np.double)

    xyz = []
    if light.polar:
        # Apply polar coordinates
        thetas = light.xrtheta.thetas
        rs = light.xrtheta.rs
        xs = light.xrtheta.xs
        for i in range(len(thetas)):
            theta = np.pi / 180 * thetas[i]
            xyz.append([xs[i], rs[i] * np.cos(theta), rs[i] * np.sin(theta)])

    else:
        # Apply cartesian coordinates
        xyz = np.array(light.xyz.coords)

    translation = np.array(xyz + origin)
    translation = torch.as_tensor(translation, dtype=torch.float32)

    light_positions0 = torch.unsqueeze(translation, dim=0)

    # light color
    light_intensities = torch.as_tensor(light.colors, ) * 1.0

    light_intensities0 = torch.unsqueeze(light_intensities, dim=0)
    light_intensities = torch.repeat_interleave(light_intensities0, 4, dim=0)

    # diffuse color
    diffuse_colors = torch.ones((4, height, width, 3), )

    # specular_colors
    specular_colors = diffuse_colors

    # shininess_coefficients
    shininess_coefficients = torch.ones((1, height, width), )

    # ambient_color
    ambient_color = torch.ones((1, 100, 3), )

    light = light

    light_positions = update_light(torch.as_tensor([0, -0.015, -0.0]),
                                   torch.as_tensor([0.5, -0.5, -0.5, -0.5]),
                                   light_positions0)
    light_positions = torch.repeat_interleave(light_positions, 4, dim=0)
    return light_positions, light_intensities, diffuse_colors


def update_light(
    cameras_pos,
    cameras_rot,
    light_positions0,
):
    '''
        update the position of light
        '''

    quat = cameras_rot
    rotation = quaternion_to_matrix(quaternion_invert(quat))
    transform = Transform3d(device="cpu").rotate(rotation[None, ]).translate(
        cameras_pos[None, ])

    light_positions = transform.transform_points(light_positions0)

    return light_positions


#========================================================================================
# phong shader model(https://github.com/google/tf_mesh_renderer)
#========================================================================================
def phong_shader(normals: torch.Tensor,
                 alphas: torch.Tensor,
                 pixel_positions: torch.Tensor,
                 light_positions: torch.Tensor,
                 light_intensities: torch.Tensor,
                 diffuse_colors=None,
                 camera_position=None,
                 specular_colors=None,
                 shininess_coefficients=None,
                 ambient_color=None) -> Union[torch.Tensor, torch.Tensor]:
    """Computes pixelwise lighting from rasterized buffers with the Phong model.
    `Args:
    normals: a 4D float32 tensor with shape [batch_size, image_height,
        image_width, 3]. The inner dimension is the world space XYZ normal for
        the corresponding pixel. Should be already normalized.
    alphas: a 3D float32 tensor with shape [batch_size, image_height,
        image_width]. The inner dimension is the alpha value (transparency)
        for the corresponding pixel.
    pixel_positions: a 4D float32 tensor with shape [batch_size, image_height,
        image_width, 3]. The inner dimension is the world space XYZ position for
        the corresponding pixel.
    light_positions: a 3D tensor with shape [batch_size, light_count, 3]. The
        XYZ position of each light in the scene. In the same coordinate space as
        pixel_positions.
    light_intensities: a 3D tensor with shape [batch_size, light_count, 3]. The
        RGB intensity values for each light. Intensities may be above one.
    diffuse_colors: a 4D float32 tensor with shape [batch_size, image_height,
        image_width, 3]. The inner dimension is the diffuse RGB coefficients at
        a pixel in the range [0, 1].
    camera_position: a 1D tensor with shape [batch_size, 3]. The XYZ camera
        position in the scene. If supplied, specular reflections will be
        computed. If not supplied, specular_colors and shininess_coefficients
        are expected to be None. In the same coordinate space as
        pixel_positions.
    specular_colors: a 4D float32 tensor with shape [batch_size, image_height,
        image_width, 3]. The inner dimension is the specular RGB coefficients at
        a pixel in the range [0, 1]. If None, assumed to be tf.zeros()
    shininess_coefficients: A 3D float32 tensor that is broadcasted to shape
        [batch_size, image_height, image_width]. The inner dimension is the
        shininess coefficient for the object at a pixel. Dimensions that are
        constant can be given length 1, so [batch_size, 1, 1] and [1, 1, 1] are
        also valid input shapes.
    ambient_color: a 2D tensor with shape [batch_size, 3]. The RGB ambient
        color, which is added to each pixel before tone mapping. If None, it is
        assumed to be tf.zeros().
    Returns:
    A 4D float32 tensor of shape [batch_size, image_height, image_width, 4]
    containing the lit RGBA color values for each image at each pixel. Colors
    are in the range [0,1].
    Raises:
    ValueError: An invalid argument to the method is detected.
    """

    batch_size, image_height, image_width = [s for s in normals.shape[:-1]]
    light_count = light_positions.shape[1]
    pixel_count = image_height * image_width
    normals = normals.view(batch_size, -1, 3)
    alphas = alphas.view(batch_size, -1, 1)
    diffuse_colors = diffuse_colors.view(batch_size, -1, 3)

    if camera_position is not None:
        specular_colors = specular_colors.view(batch_size, -1, 3)

    output_colors = torch.zeros([batch_size, image_height * image_width, 3], )
    if ambient_color is not None:
        ambient_reshaped = torch.unsqueeze(ambient_color, axis=1)
        output_colors = torch.add(output_colors,
                                  ambient_reshaped * diffuse_colors)

    # Diffuse component
    pixel_positions = pixel_positions.view(batch_size, -1, 3)
    per_light_pixel_positions = torch.stack(
        [pixel_positions] * light_count,
        axis=1)  # [batch_size, light_count, pixel_count, 3]
    directions_to_lights = torch.nn.functional.normalize(
        torch.unsqueeze(light_positions, axis=2) - per_light_pixel_positions,
        dim=3)  # [batch_size, light_count, pixel_count, 3]

    # The specular component should only contribute when the light and normal
    # face one another (i.e. the dot product is nonnegative):
    normals_dot_lights = torch.clamp(
        torch.sum(torch.unsqueeze(normals, axis=1) * directions_to_lights,
                  axis=3), 0.0, 1.0)  # [batch_size, light_count, pixel_count]

    diffuse_output = torch.unsqueeze(diffuse_colors, axis=1) * torch.unsqueeze(
        normals_dot_lights, axis=3) * torch.unsqueeze(light_intensities,
                                                      axis=2)

    diffuse_output = torch.sum(diffuse_output,
                               axis=1)  # [batch_size, pixel_count, 3]
    output_colors = torch.add(output_colors, diffuse_output)

    # Specular component
    if camera_position is not None:
        camera_position = camera_position.view(batch_size, 1, 3)
        mirror_reflection_direction = torch.nn.functional.normalize(
            2.0 * torch.unsqueeze(normals_dot_lights, axis=3) *
            torch.unsqueeze(normals, axis=1) - directions_to_lights,
            dim=3)
        direction_to_camera = torch.nn.functional.normalize(camera_position -
                                                            pixel_positions,
                                                            dim=2)
        reflection_direction_dot_camera_direction = torch.sum(
            torch.unsqueeze(direction_to_camera, axis=1) *
            mirror_reflection_direction,
            axis=3)
        # The specular component should only contribute when the reflection is
        # external:
        reflection_direction_dot_camera_direction = torch.clamp(
            torch.nn.functional.normalize(
                reflection_direction_dot_camera_direction, dim=2), 0.0, 1.0)
        # The specular component should also only contribute when the diffuse
        # component contributes:
        reflection_direction_dot_camera_direction = torch.where(
            normals_dot_lights != 0.0,
            reflection_direction_dot_camera_direction,
            torch.zeros_like(reflection_direction_dot_camera_direction,
                             dtype=torch.float32))
        # Reshape to support broadcasting the shininess coefficient, which rarely
        # varies per-vertex:
        reflection_direction_dot_camera_direction = reflection_direction_dot_camera_direction.view(
            batch_size, light_count, image_height, image_width)
        shininess_coefficients = torch.unsqueeze(shininess_coefficients,
                                                 axis=1)
        specularity = torch.pow(reflection_direction_dot_camera_direction,
                                shininess_coefficients).view(
                                    batch_size, light_count, pixel_count, 1)
        specular_output = torch.unsqueeze(
            specular_colors, axis=1) * specularity * torch.unsqueeze(
                light_intensities, axis=2)
        specular_output = torch.sum(specular_output, axis=1)
        output_colors = torch.add(output_colors, specular_output)
    rgb_images = output_colors.view(batch_size, image_height, image_width, 3)

    alpha_images = alphas.view(batch_size, image_height, image_width, 1)
    valid_rgb_values = torch.cat(3 * [alpha_images > 0.5], axis=3)
    rgb_images = torch.where(valid_rgb_values, rgb_images,
                             torch.zeros_like(rgb_images, dtype=torch.float32))
    return torch.cat([rgb_images, alpha_images], axis=3)


def fetch_tactile_image(pixel_normals, pixel_coords):
    light_positions, light_intensities, diffuse_colors = init_light()
    alphas = torch.ones((4, height, width)) * 1.0
    batch_size = 4

    rgb = phong_shader(
        pixel_normals[:, :, :, :3],
        alphas,
        pixel_coords[:, :, :, :3],
        light_positions,
        light_intensities.reshape(-1, 3, 3),
        diffuse_colors[:batch_size],
        # cameras_center,
        # specular_colors[:batch_size],
        # shininess_coefficients[:batch_size],
        # ambient_color[:batch_size]
    )

    updateGUI(rgb[:, :, :, :3].numpy(), depths=pixel_coords[:, :, :, 2])

    return rgb
