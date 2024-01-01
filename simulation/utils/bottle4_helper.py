from http.client import PARTIAL_CONTENT
import json
from pathlib import Path
from xml.parsers.expat import model

import numpy as np
import sapien.core as sapien
from scipy.spatial.transform import Rotation as R
import pdb
import trimesh
import xml.etree.ElementTree as ET
from hand_teleop.utils.facucet_setting import FAUCET_SCALE, load_faucet, build_articulation_faucet

BOTTLE_LINK_ORFER1 = ["base", "link_1", "link_0_helper", "link_0", "link_0_0"]
BOTTLE_LINK_ORFER2 = ["base", "link_1", "link_0_helper", "link_0"]
BOTTLE3_SCALE = 0.1
DOWN_HAND_POSE = {
    "cylinder": [0.55, 0, -0.03],
    "cone": [0.55, 0, -0.02],
    "icosphere": [0.55, 0, -0.02],
    "capsule32": [0.55, 0, -0.05],
    "torus": [0.55, 0, -0.03],
    "uptorus": [0.55, 0, 0.03]
}
SCALE_FACTOR = {
    "cylinder": 1,
    "cone": 2,
    "icosphere": 3,
    "capsule": 1.5,
    "torus": 2.,
    "uptorus": 2
}
HAND_SCALE_FACTOR = {
    "front": 1.2,
    "down": {
        "cylinder_3": 1.9,
        "cylinder_4": 1.9,
        "cylinder_5": 1.8,
        "cylinder_6": 1.8,
        "cylinder_7": 1.6,
        "cylinder_8": 1.6,
        "cylinder_9": 1.6,
        "cylinder_100": 1.2,
        "capsule": 1.6,
        "torus": 1.8,
        "uptorus": 2.5
    }
}
#=============================================================================================
#==================================== bottle4 ================================================
#=============================================================================================


def create_texture(renderer, index: int = 0):

    if renderer:
        color = [[233, 119, 119, 255], [195, 248, 255, 255]]
        visual_material = renderer.create_material()
        visual_material.set_metallic(0.0)
        visual_material.set_specular(0.3)
        visual_material.set_base_color(np.array(color[index]) / 255)
        visual_material.set_roughness(0.3)
        return visual_material
    else:
        return None


def create_friction_material(scene):

    mat = scene.create_physical_material(float(np.random.uniform(0.5, 1.0)),
                                         float(np.random.uniform(0.5, 1.0)), 0)
    # mat = scene.create_physical_material(1,
    #                                      1, 0)

    # mat.set_static_friction(float(np.random.uniform(0., 0.2)))
    # mat.set_dynamic_friction(np.random.uniform(0., 0.2))
    return mat


def load_bottle_polygon1(scene, model_name, scale, renderer):

    x_scale = scale[0] / 1.8

    cap_scale = scale[1] / 1
    bottle_scale = scale[2] / 1

    cap_scales = [x_scale[0], x_scale[0], cap_scale[0]]
    bottle_scales = [x_scale[0], x_scale[0], bottle_scale[0]]

    builder: sapien.ArticulationBuilder = scene.create_articulation_builder()

    #base link and fix joint
    base_link = builder.create_link_builder()
    base_link.set_name("base")
    base_link.set_joint_properties(joint_type="fixed",
                                   limits=[],
                                   pose_in_parent=sapien.Pose([0, 0, 0],
                                                              [1, 0, 0, 0]))

    # bottle link and parents link
    link_bottle = builder.create_link_builder()
    link_bottle.add_multiple_collisions_from_file(
        "./assets/partnet-mobility-dataset/bottle4/cylinder_%d.stl" %
        int(model_name),
        scale=bottle_scales,
        pose=sapien.Pose([0, 0, 0], [0.707, 0, 0.707, 0]))
    bottle_texture = create_texture(renderer, 0)  #cap textuew
    link_bottle.add_visual_from_file(
        "./assets/partnet-mobility-dataset/bottle4/cylinder_%d.stl" %
        int(model_name),
        scale=bottle_scales,
        pose=sapien.Pose([0, 0, 0], [0.707, 0, 0.707, 0]))

    link_bottle.set_name("bottle")
    link_bottle.set_parent(base_link.get_index())

    #link helper for prismatic joints
    # link_helper = builder.create_link_builder()
    # link_helper.set_name("link_helper")
    # link_helper.set_joint_properties(
    #     joint_type="prismatic",
    #     limits=[[-0.2, 0.00]],
    #     pose_in_child=sapien.Pose([0, 0, 0], [0.707, 0, 0, 0.707]),
    #     pose_in_parent=sapien.Pose([0, 0, 0], [0.707, 0, 0, 0.707]))

    # cap link and revolute joints
    link_cap = builder.create_link_builder()

    link_cap.add_multiple_collisions_from_file(
        "./assets/partnet-mobility-dataset/bottle4/cylinder_%d.stl" %
        int(model_name),
        scale=cap_scales)
    cap_texture = create_texture(renderer, 1)  #cap textuew
    link_cap.add_visual_from_file(
        "./assets/partnet-mobility-dataset/bottle4/cylinder_%d.stl" %
        int(model_name),
        scale=cap_scales,
        material=cap_texture)
    link_cap.set_name("cap")

    link_cap.set_parent(link_bottle.get_index())
    # link_helper.set_parent(link_bottle.get_index())
    link_cap.set_joint_properties(
        joint_type="revolute",
        limits=[[-2 * np.pi, 2 * np.pi]],
        pose_in_child=sapien.Pose([0, 0, 0], [0.707, 0, 0.707, 0]),
        pose_in_parent=sapien.Pose(
            [0, 0, 0.5 * bottle_scale[0] + 0.5 * cap_scale[0] + 0.01],
            [0.707, 0, 0.707, 0]))

    for link in builder.get_link_builders():
        link.set_collision_groups(1, 1, 4, 4)
    object = builder.build(fix_root_link=True)

    return object, cap_scales


#============================================== Another way for bottle creation =======================================


def calculate_length(points):

    x_min, x_max = np.min(points[:, 0]), np.max(points[:, 0])
    y_min, y_max = np.min(points[:, 1]), np.max(points[:, 1])
    z_min, z_max = np.min(points[:, 2]), np.max(points[:, 2])

    return (x_max - x_min), (y_max - y_min), (
        z_max -
        z_min), (x_min + x_max) / 2, (y_min + y_max) / 2, (z_min + z_max) / 2


def load_mesh(files, name):
    points = []

    for file in files:
        path = file
        mesh = trimesh.load(path)

        if not isinstance(mesh, trimesh.Trimesh):

            for msh_key in mesh.geometry.keys():

                msh = mesh.geometry[msh_key]
                points.append(msh.vertices)

        else:
            point = mesh.vertices
            points.append(point)
    points = np.concatenate(points, axis=0)

    x_length, y_length, z_length, x_center, y_center, z_center = calculate_length(
        points)

    return x_length, y_length, z_length, x_center, y_center, z_center


# def save_bbox(dict, link_name, name):

#     x_length, y_length, z_length, x_center, y_center, z_center = load_mesh(
#         dict['link'][link_name]['geometry'], name)
#     dict['link'][link_name]['length'] = [x_length, y_length, z_length]
#     dict['link'][link_name]['center'] = [x_center, y_center, z_center]

#     return dict


def parse_urdf(file, dict, name):

    dict["link"] = {}
    dict["joint"] = {}
    tree = ET.parse(file)
    root = tree.getroot()
    for child in root:

        if child.tag == "joint":
            dict["joint"][child.attrib['name']] = {}
            dict["joint"][child.attrib['name']]['type'] = child.attrib['type']
            for c in child:
                if c.tag in ["origin", "axis"]:

                    if c.tag not in dict["joint"][child.attrib['name']].keys():
                        dict["joint"][child.attrib['name']][c.tag] = {}

                    dict["joint"][child.attrib['name']][
                        c.tag]['xyz'] = c.attrib['xyz']

                    if "rpy" in c.attrib.keys():
                        dict["joint"][child.attrib['name']][
                            c.tag]['rpy'] = np.array(
                                c.attrib['rpy'].split()).astype(np.float32)
                    else:
                        dict["joint"][child.attrib['name']][
                            c.tag]['rpy'] = np.array([1, 0, 0, 0])

                elif c.tag == "limit":
                    dict["joint"][child.attrib['name']][c.tag] = [
                        float(c.attrib['lower']),
                        float(c.attrib['upper'])
                    ]
                else:
                    dict["joint"][child.attrib['name']][
                        c.tag] = c.attrib['link']

        else:

            if child.attrib['name'] not in dict['link'].keys():
                dict['link'][child.attrib['name']] = {}

            for c in child:

                if c.tag in ["visual"]:
                    continue

                for cc in c:

                    if cc.tag in ["origin"]:

                        if 'origin' not in dict['link'][
                                child.attrib['name']].keys():
                            dict['link'][child.attrib['name']]["origin"] = {}

                        dict['link'][
                            child.attrib['name']]["origin"]['xyz'] = np.array(
                                cc.attrib['xyz'].split()).astype(np.float32)
                        if "rpy" in cc.attrib.keys():
                            dict['link'][child.attrib['name']]["origin"][
                                'rpy'] = np.array(
                                    cc.attrib['rpy'].split()).astype(
                                        np.float32)
                        else:
                            dict['link'][child.attrib['name']]["origin"][
                                'rpy'] = np.array([1, 0, 0, 0])

                    for ccc in cc:
                        if "filename" in ccc.attrib.keys():

                            if "geometry" not in dict['link'][
                                    child.attrib['name']].keys():
                                dict['link'][
                                    child.attrib['name']]['geometry'] = []

                                # print(child.tag, child.attrib["name"])
                                # print(c.tag, ccc.attrib["filename"])
                            dict['link'][
                                child.attrib['name']]['geometry'].append(
                                    "./assets/partnet-mobility-dataset/bottle/"
                                    + name + "/" + ccc.attrib["filename"])

    return dict


def build_articulation(dict,
                       scene,
                       model_name,
                       scale=None,
                       renderer=None,
                       down=False):

    builder: sapien.ArticulationBuilder = scene.create_articulation_builder()

    link_dicts = dict["link"]
    links = {}

    if "cylinder" not in model_name:
        bottle_index = 100  # use cylinder as base

    else:
        bottle_index = model_name.split("_")[1]

    if "cylinder" in model_name:
        cap_x_scale = scale[0] / sections_length["capsule"][0]
        bottle_x_scale = 0.5 * scale[0] / sections_length[
            "cylinder_" + str(bottle_index)][0]  #height
    else:
        cap_x_scale = scale[0] / sections_length[str(model_name)][0]  #x
        bottle_x_scale = scale[0] / sections_length[
            "cylinder_" + str(bottle_index)][0]  #height

    cap_scale = scale[1] / sections_length[str(model_name)][2]  #height
    bottle_scale = scale[2] / sections_length["cylinder_" +
                                              str(bottle_index)][2]  #height
    cap_scales = [cap_x_scale, cap_x_scale, cap_scale]
    bottle_scales = [bottle_x_scale * 0.4, bottle_x_scale * 0.4, bottle_scale]

    if "cylinder" in model_name:
        height_factor = 1  # height scale factor for acconmendation
    else:
        height_factor = 0

        1

    for name in BOTTLE_LINK_ORFER1:
        link = builder.create_link_builder()
        link.set_name(name)

        if name in ["link_1"]:
            scales = bottle_scales
        else:
            scales = cap_scales

        if bool(link_dicts[name]):

            if name in ["link_1"]:

                if "cylinder" not in model_name:
                    cylinder_inedx = 100

                else:
                    cylinder_inedx = int(model_name.split("_")[1])

                link.add_multiple_collisions_from_file(
                    "./assets/partnet-mobility-dataset/bottle4/cylinder_%d" %
                    cylinder_inedx + ".stl",
                    pose=sapien.Pose([0, 0, 0], [1, 0, 0, 0]),
                    scale=scales)

                # bottle_texture = create_texture(renderer, 0)  #bottle textuew
                link.add_visual_from_file(
                    "./assets/partnet-mobility-dataset/bottle4/cylinder_%d" %
                    cylinder_inedx + ".stl",
                    pose=sapien.Pose([0, 0, 0], [1, 0, 0, 0]),
                    scale=scales,
                    material=None)

            elif name in ["link_0"]:

                if down:
                    links[name] = link
                    continue

                # else:
                # friction_material = create_friction_material(scene)
                # pdb.set_trace()
                temp_scales = [
                    cap_x_scale * 0.9, cap_x_scale * 0.9, cap_scale * 1.1
                ]
                # print("temp", cap_scales, scales)
                link.add_multiple_collisions_from_file(
                    "./assets/partnet-mobility-dataset/bottle4/" +
                    (model_name) + ".stl",
                    pose=sapien.Pose([0, 0, 0], [1, 0, 0, 0]),
                    scale=temp_scales)

                # cap_texture = create_texture(renderer, 0)  #bottle textuew
                link.add_visual_from_file(
                    "./assets/partnet-mobility-dataset/bottle4/" +
                    (model_name) + ".stl",
                    pose=sapien.Pose([0, 0, 0], [1, 0, 0, 0]),
                    scale=temp_scales,
                    material=None)

            elif name in ["link_0_0"]:
                # print("temp", scales)

                link.add_multiple_collisions_from_file(
                    "./assets/partnet-mobility-dataset/bottle4/" +
                    (model_name) + ".stl",
                    pose=sapien.Pose([0, 0, 0], [1, 0, 0, 0]),
                    scale=scales)
                # cap_texture = create_texture(renderer, 1)  #bottle textuew
                link.add_visual_from_file(
                    "./assets/partnet-mobility-dataset/bottle4/" +
                    (model_name) + ".stl",
                    pose=sapien.Pose([0, 0, 0], [1, 0, 0, 0]),
                    scale=scales,
                    material=None)

        links[name] = link

    joint_dicts = dict["joint"]

    continuous_joint_info = {}
    for name in joint_dicts.keys():

        child_name = joint_dicts[name]['child']
        parent_name = joint_dicts[name]['parent']

        if joint_dicts[name]['type'] == "fixed":
            links[child_name].set_joint_properties(
                joint_type="fixed",
                limits=[],
                pose_in_parent=sapien.Pose([0, 0, 0], [1, 0, 0, 0]))

        if joint_dicts[name]['type'] == "prismatic":
            links[continuous_joint_info["child_name"].get_name(
            )].set_joint_properties(
                joint_type="revolute",
                limits=[[-2 * np.pi, 2 * np.pi]],
                pose_in_child=sapien.Pose([0, 0, 0], [
                    0.707,
                    0,
                    0.707,
                    0,
                ]),
                pose_in_parent=sapien.Pose([
                    0, 0, 0.5 * bottle_scale + 0.5 * cap_scale * height_factor
                ], [0.707, 0, 0.707, 0]))

            links[child_name].set_joint_properties(
                joint_type="prismatic",
                limits=[[-0.0, 0.00]],
                pose_in_child=sapien.Pose([0, 0, 0], [0.707, 0, 0.707, 0]),
                pose_in_parent=sapien.Pose([0, 0, 0], [0.707, 0, 0.707, 0]))

        if joint_dicts[name]['type'] == "continuous":
            continuous_joint_info["joint_type"] = "revolute"
            continuous_joint_info["child_name"] = links[child_name]
            continuous_joint_info["limits"] = [[-2 * np.pi, 2 * np.pi]]
            continuous_joint_info["xyz"] = np.array(
                joint_dicts[name]['origin']['xyz'].split()).astype(
                    np.float32)[[0, 2, 1]]

        links[child_name].set_parent(links[parent_name].get_index())

    for link in builder.get_link_builders():
        link.set_collision_groups(1, 1, 4, 4)

    object = builder.build(fix_root_link=True)
    render_scale = np.array(cap_scales)  #[[0, 2, 1]]

    return object, render_scale


def load_bottle_polygon(scene, model_name, scale, renderer, down):
    object, render_scale = build_articulation(bottle3_dict["4084"], scene,
                                              model_name, scale, renderer,
                                              down)

    return object, render_scale


bottle3_dict = {
    '4084': {
        'link': {
            'base': {},
            'link_0_0': {
                'origin': {
                    'xyz': np.array([0.00229109, 0., 0.01429],
                                    dtype=np.float32),
                    'rpy': np.array([1, 0, 0, 0])
                },
                'geometry': [
                    './assets/partnet-mobility-dataset/bottle/4084/textured_objs/original-1.obj'
                ]
            },
            'link_0': {
                'origin': {
                    'xyz': np.array([0.00229109, 0., 0.01429],
                                    dtype=np.float32),
                    'rpy': np.array([1, 0, 0, 0])
                },
                'geometry': [
                    './assets/partnet-mobility-dataset/bottle/4084/split_mesh/cylinder.stl'
                ]
            },
            'link_1': {
                'origin': {
                    'xyz': np.array([0., 0., 0.], dtype=np.float32),
                    'rpy': np.array([1, 0, 0, 0])
                },
                'geometry': [
                    './assets/partnet-mobility-dataset/bottle/4084/textured_objs/new-1.obj',
                    './assets/partnet-mobility-dataset/bottle/4084/textured_objs/new-3.obj',
                    './assets/partnet-mobility-dataset/bottle/4084/textured_objs/original-4.obj',
                    './assets/partnet-mobility-dataset/bottle/4084/textured_objs/original-39.obj',
                    './assets/partnet-mobility-dataset/bottle/4084/textured_objs/original-18.obj',
                    './assets/partnet-mobility-dataset/bottle/4084/textured_objs/original-32.obj',
                    './assets/partnet-mobility-dataset/bottle/4084/textured_objs/original-25.obj',
                    './assets/partnet-mobility-dataset/bottle/4084/textured_objs/original-11.obj'
                ]
            },
            'link_0_helper': {}
        },
        'joint': {
            'joint_0_0': {
                'type': 'fixed',
                'origin': {
                    'xyz': '0.0 0 0.0',
                    'rpy': np.array([1, 0, 0, 0])
                },
                'axis': {
                    'xyz': '0 1 0',
                    'rpy': np.array([1, 0, 0, 0])
                },
                'child': 'link_0_0',
                'parent': 'link_0'
            },
            'joint_0': {
                'type': 'continuous',
                'origin': {
                    'xyz': '0.0 0 0.0',
                    'rpy': np.array([1, 0, 0, 0])
                },
                'axis': {
                    'xyz': '0 1 0',
                    'rpy': np.array([1, 0, 0, 0])
                },
                'child': 'link_0',
                'parent': 'link_0_helper'
            },
            'joint_1': {
                'type': 'fixed',
                'origin': {
                    'xyz': '0 0 0',
                    'rpy': np.array([1.5707964, 0., -1.5707964],
                                    dtype=np.float32)
                },
                'child': 'link_1',
                'parent': 'base'
            },
            'joint_2': {
                'type': 'prismatic',
                'origin': {
                    'xyz': '-0.002291089456653725 0 -0.014289999999999997',
                    'rpy': np.array([1, 0, 0, 0])
                },
                'axis': {
                    'xyz': '0 1 0',
                    'rpy': np.array([1, 0, 0, 0])
                },
                'child': 'link_0_helper',
                'parent': 'link_1',
                'limit': [-0.06400000000000006, 0.2639999999999998]
            }
        }
    }
}
# for model_name in ["4084"]:

#     if model_name in bottle3_dict.keys():
#         continue
#     bottle3_dict[model_name] = {}
#     bottle3_dict[model_name] = parse_urdf(
#         "../sapien_task/assets/partnet-mobility-dataset/bottle/" + model_name +
#         "/mobility.urdf", bottle3_dict[model_name], model_name)

# sections_length = {}
# # for i in range(0, 10):

# #     sections_length["icosphere_" + str(i)] = load_mesh([
# #         "./assets/partnet-mobility-dataset/bottle4/cone_%d" % (i) + ".stl"
# #     ], None)
# i = 32
# sections_length = {}
# for i in range(2, 11):
#     sections_length["uptorus_" + str(i)] = load_mesh([
#         "./assets/partnet-mobility-dataset/bottle4/uptorus_%d" % (i) + ".stl"
#     ], None)
# pdb.set_trace()

sections_length = {
    'cylinder_3': (1.5, 1.7320507764816284, 1.0, 0.25, 0.0, 0.0),
    'cylinder_4': (2.0, 2.0, 1.0, 0.0, 0.0, 0.0),
    'cylinder_5':
    (1.80901700258255, 1.9021130800247192, 1.0, 0.09549149870872498, 0.0, 0.0),
    'cylinder_6': (2.0, 1.7320507764816284, 1.0, 0.0, 0.0, 0.0),
    'cylinder_7': (1.900968849658966, 1.9498558044433594, 1.0,
                   0.04951557517051697, 0.0, 0.0),
    'cylinder_8': (2.0, 2.0, 1.0, 0.0, 0.0, 0.0),
    'cylinder_9': (1.9396926164627075, 1.9696154594421387, 1.0,
                   0.03015369176864624, 0.0, 0.0),
    'cylinder_100': (2.0, 2.0, 1.0, 0.0, 0.0, 0.0),
    'cone_3': (1.5, 1.7320507764816284, 1.0, 0.25, 0.0, 0.5),
    'cone_4': (2.0, 2.0, 1.0, 0.0, 0.0, 0.5),
    'cone_5':
    (1.80901700258255, 1.9021130800247192, 1.0, 0.09549149870872498, 0.0, 0.5),
    'cone_6': (2.0, 1.7320507764816284, 1.0, 0.0, 0.0, 0.5),
    'cone_7': (1.900968849658966, 1.9498558044433594, 1.0, 0.04951557517051697,
               0.0, 0.5),
    'cone_8': (2.0, 2.0, 1.0, 0.0, 0.0, 0.5),
    'cone_9': (1.9396926164627075, 1.9696154594421387, 1.0,
               0.03015369176864624, 0.0, 0.5),
    'cone_100': (2.0, 2.0, 1.0, 0.0, 0.0, 0.5),
    'icosphere_0': (1.9987569451332092, 1.9993783235549927, 0.9999999999999999,
                    0.0006215274333953857, 0.0, 0.5),
    'icosphere_1': (1.9987569451332092, 1.9993783235549927, 0.9969173073768615,
                    0.0006215274333953857, 0.0, 0.49845865368843084),
    'icosphere_2': (1.9987569451332092, 1.9993783235549927, 0.9876883625984191,
                    0.0006215274333953857, 0.0, 0.49384418129920965),
    'icosphere_3': (1.9987569451332092, 1.9993783235549927, 0.9723699092864989,
                    0.0006215274333953857, 0.0, 0.48618495464324957),
    'icosphere_4': (1.9987569451332092, 1.9993783235549927, 0.9510565400123595,
                    0.0006215274333953857, 0.0, 0.47552827000617987),
    'icosphere_5': (1.9987569451332092, 1.9993783235549927, 0.9238795042037963,
                    0.0006215274333953857, 0.0, 0.46193975210189825),
    'icosphere_6': (1.9987569451332092, 1.9993783235549927, 0.8910065293312072,
                    0.0006215274333953857, 0.0, 0.4455032646656037),
    'icosphere_7': (1.9987569451332092, 1.9993783235549927, 0.852640151977539,
                    0.0006215274333953857, 0.0, 0.4263200759887696),
    'icosphere_8': (1.9987569451332092, 1.9993783235549927, 0.8090170025825499,
                    0.0006215274333953857, 0.0, 0.4045085012912751),
    'icosphere_9': (1.9987569451332092, 1.9993783235549927, 0.7604059576988219,
                    0.0006215274333953857, 0.0, 0.38020297884941107),
    'capsule': (1.9987569451332092, 1.9993783235549927, 3.0,
                0.0006215274333953857, 0.0, 0.5),
    'capsule_2': (2.0, 0.7997512817382812, 0.6000000089406967, 0.0, 0.0,
                  0.10000000149011612),
    'capsule_3':
    (1.9987568855285645, 1.7685257196426392, 0.6000000089406967,
     0.0006215572357177734, 0.0031935572624206543, 0.10000000149011612),
    'capsule_4': (2.0, 2.0, 0.6000000089406967, 0.0, 0.0, 0.10000000149011612),
    'capsule_5':
    (1.9987568855285645, 1.9070920944213867, 0.6000000089406967,
     0.0006215572357177734, 0.0024895071983337402, 0.10000000149011612),
    'capsule_6': (2.0, 1.7749128341674805, 0.6000000089406967, 0.0, 0.0,
                  0.10000000149011612),
    'capsule_7':
    (1.9987568855285645, 1.9530805945396423, 0.6000000089406967,
     0.0006215572357177734, -0.0016123950481414795, 0.10000000149011612),
    'capsule_8': (2.0, 2.0, 0.6000000089406967, 0.0, 0.0, 0.10000000149011612),
    'capsule_9': (1.9987568855285645, 1.971853494644165, 0.6000000089406967,
                  0.0006215572357177734, 0.0011190176010131836,
                  0.10000000149011612),
    'capsule_10': (2.0, 1.9120711088180542, 0.6000000089406967, 0.0, 0.0,
                   0.10000000149011612),
    'capsule_11': (1.9987568855285645, 1.9812499284744263, 0.6000000089406967,
                   0.0006215572357177734, -0.000803530216217041,
                   0.10000000149011612),
    'capsule_12': (2.0, 2.0, 0.6000000089406967, 0.0, 0.0,
                   0.10000000149011612),
    'capsule_13': (1.9987568855285645, 1.9865869879722595, 0.6000000089406967,
                   0.0006215572357177734, 0.0005846321582794189,
                   0.10000000149011612),
    'thincapsule_1': (1.9987568855285645, 0.7987571656703949,
                      0.6000000089406967, 0.0006215572357177734,
                      -0.0004970580339431763, 0.10000000149011612),
    'thincapsule_2': (1.9987568855285645, 1.9987568855285645,
                      0.6000000089406967, 0.0006215572357177734,
                      -0.0006215572357177734, 0.10000000149011612),
    'thincapsule_3': (1.9987568855285645, 1.7728199362754822,
                      0.6000000089406967, 0.0006215572357177734,
                      0.0010464489459991455, 0.10000000149011612),
    'thincapsule_4': (1.9987568855285645, 1.9987568855285645,
                      0.6000000089406967, 0.0006215572357177734,
                      -0.0006215572357177734, 0.10000000149011612),
    'thincapsule_5': (1.9987568855285645, 1.9070920944213867,
                      0.6000000089406967, 0.0006215572357177734,
                      0.0024895071983337402, 0.10000000149011612),
    'thincapsule_6': (1.9987568855285645, 1.9987568855285645,
                      0.6000000089406967, 0.0006215572357177734,
                      -0.0006215572357177734, 0.10000000149011612),
    'thincapsule_7': (1.9987568855285645, 1.9530805945396423,
                      0.6000000089406967, 0.0006215572357177734,
                      0.0016123950481414795, 0.10000000149011612),
    'thincapsule2_1': (1.9987568855285645, 0.39937858283519745,
                       0.6000000089406967, 0.0006215572357177734,
                       -0.00024852901697158813, 0.10000000149011612),
    'thincapsule2_2': (1.9987568855285645, 1.9987568855285645,
                       0.6000000089406967, 0.0006215572357177734,
                       -0.0006215572357177734, 0.10000000149011612),
    'thincapsule2_3': (1.9987568855285645, 1.7359588146209717,
                       0.6000000089406967, 0.0006215572357177734,
                       0.001954019069671631, 0.10000000149011612),
    'thincapsule2_4': (1.9987568855285645, 1.9987568855285645,
                       0.6000000089406967, 0.0006215572357177734,
                       -0.0006215572357177734, 0.10000000149011612),
    'thincapsule2_5': (1.9987568855285645, 1.9040114283561707,
                       0.6000000089406967, 0.0006215572357177734,
                       0.000949174165725708, 0.10000000149011612),
    'thincapsule2_6': (1.9987568855285645, 1.9987568855285645,
                       0.6000000089406967, 0.0006215572357177734,
                       -0.0006215572357177734, 0.10000000149011612),
    'thincapsule2_7': (1.9987568855285645, 1.9508622288703918,
                       0.6000000089406967, 0.0006215572357177734,
                       0.0005032122135162354, 0.10000000149011612),
    'thincapsule2_8': (1.9987568855285645, 1.9987568855285645,
                       0.6000000089406967, 0.0006215572357177734,
                       -0.0006215572357177734, 0.10000000149011612),
    'thincapsule2_9': (1.9987568855285645, 1.9701223969459534,
                       0.6000000089406967, 0.0006215572357177734,
                       0.00025346875190734863, 0.10000000149011612),
    'thincapsule2_10': (1.9987568855285645, 1.9987568855285645,
                        0.6000000089406967, 0.0006215572357177734,
                        -0.0006215572357177734, 0.10000000149011612),
    'thincapsule3_1': (1.9987568855285645, 0.19968929141759872,
                       0.6000000089406967, 0.0006215572357177734,
                       -0.00012426450848579407, 0.10000000149011612),
    'thincapsule3_2': (1.9987568855285645, 1.9987568855285645,
                       0.6000000089406967, 0.0006215572357177734,
                       -0.0006215572357177734, 0.10000000149011612),
    'thincapsule3_3': (1.9987568855285645, 1.7334665060043335,
                       0.6000000089406967, 0.0006215572357177734,
                       0.0007078647613525391, 0.10000000149011612),
    'thincapsule3_4': (1.9987568855285645, 1.9987568855285645,
                       0.6000000089406967, 0.0006215572357177734,
                       -0.0006215572357177734, 0.10000000149011612),
    'thincapsule3_5': (1.9987568855285645, 1.902471125125885,
                       0.6000000089406967, 0.0006215572357177734,
                       0.00017902255058288574, 0.10000000149011612),
    'thincapsule3_6': (1.9987568855285645, 1.9987568855285645,
                       0.6000000089406967, 0.0006215572357177734,
                       -0.0006215572357177734, 0.10000000149011612),
    'thincapsule3_7': (1.9987568855285645, 1.9497530460357666,
                       0.6000000089406967, 0.0006215572357177734,
                       -5.137920379638672e-05, 0.10000000149011612),
    'thincapsule3_8': (1.9987568855285645, 1.9987568855285645,
                       0.6000000089406967, 0.0006215572357177734,
                       -0.0006215572357177734, 0.10000000149011612),
    'thincapsule3_9': (1.9987568855285645, 1.9692568182945251,
                       0.6000000089406967, 0.0006215572357177734,
                       -0.0001793205738067627, 0.10000000149011612),
    'thincapsule3_10': (1.9987568855285645, 1.9987568855285645,
                        0.6000000089406967, 0.0006215572357177734,
                        -0.0006215572357177734, 0.10000000149011612),
    'thincapsule3_11': (1.9987568855285645, 1.9791218042373657,
                        0.6000000089406967, 0.0006215572357177734,
                        -0.0002605319023132324, 0.10000000149011612),
    'thincapsule3_12': (1.9987568855285645, 1.9987568855285645,
                        0.6000000089406967, 0.0006215572357177734,
                        -0.0006215572357177734, 0.10000000149011612),
    'thincapsule3_13': (1.9987568855285645, 1.984784483909607,
                        0.6000000089406967, 0.0006215572357177734,
                        -0.000316619873046875, 0.10000000149011612),
    'thincapsule3_14': (1.9987568855285645, 1.9987568855285645,
                        0.6000000089406967, 0.0006215572357177734,
                        -0.0006215572357177734, 0.10000000149011612),
    'thincapsule3_15': (1.9987568855285645, 1.9883285164833069,
                        0.6000000089406967, 0.0006215572357177734,
                        -0.00035765767097473145, 0.10000000149011612),
    'thincapsule3_16': (1.9987568855285645, 1.9987568855285645,
                        0.6000000089406967, 0.0006215572357177734,
                        -0.0006215572357177734, 0.10000000149011612),
    'torus_2': (3.6000001430511475, 4.176991939544678, 0.6000000238418579, 0.0,
                0.0, 0.15000000596046448),
    'torus_3': (3.4499998092651367, 3.431100368499756, 0.9000000059604645, 0.0,
                0.0, 0.29999999701976776),
    'torus_4': (3.6000001430511475, 3.5802788734436035, 1.1999999582767487,
                0.0, 0.0, 0.44999997317790985),
    'torus_5': (3.75, 3.729456901550293, 1.5000000298023224, 0.0, 0.0,
                0.6000000089406967),
    'torus_6': (3.8999998569488525, 3.8786351680755615, 1.7999999821186066,
                0.0, 0.0, 0.7499999850988388),
    'torus_7': (4.050000190734863, 4.02781343460083, 2.0999999344348907, 0.0,
                0.0, 0.8999999612569809),
    'torus_8': (4.199999809265137, 4.176991939544678, 2.4000000059604645, 0.0,
                0.0, 1.0499999970197678),
    'torus_9': (4.350000381469727, 4.326170444488525, 2.7000001966953278, 0.0,
                0.0, 1.2000000923871994),
    'torus_10': (4.5, 4.475348472595215, 3.000000149011612, 0.0, 0.0,
                 1.3500000685453415),
    'torus_11': (4.649999618530273, 4.624526500701904, 3.300000101327896, 0.0,
                 0.0, 1.5000000447034836),
    'uptorus_2': (2.9652243852615356, 3.31594717502594, 0.6052088737487793,
                  0.002376735210418701, -0.021822869777679443,
                  0.14898431301116943),
    'uptorus_3': (2.9295743703842163, 2.8033119440078735, 0.9040783941745758,
                  -0.015448272228240967, 0.0786709189414978,
                  0.29954956471920013),
    'uptorus_4': (2.9039262533187866, 2.850770592689514, 1.2054351270198822,
                  -0.002624213695526123, 0.05494159460067749,
                  0.45022793114185333),
    'uptorus_5': (2.9039262533187866, 2.850770592689514, 1.5061903297901154,
                  -0.002624213695526123, 0.05494159460067749,
                  0.5998504012823105),
    'uptorus_6': (2.9039262533187866, 2.850770592689514, 1.8061031848192215,
                  -0.002624213695526123, 0.05494159460067749,
                  0.7498939260840416),
    'uptorus_7': (2.9039262533187866, 2.850770592689514, 2.105881243944168,
                  -0.002624213695526123, 0.05494159460067749,
                  0.9000048488378525),
    'uptorus_8': (2.9039264917373657, 2.8507708311080933, 2.4058423191308975,
                  -0.002624213695526123, 0.05494159460067749,
                  1.050024263560772),
    'uptorus_9': (2.9039264917373657, 2.8507708311080933, 2.7046695351600647,
                  -0.002624213695526123, 0.05494159460067749,
                  1.2006108462810516),
    'uptorus_10': (2.9039264917373657, 2.8507708311080933, 3.0044853389263153,
                   -0.002624213695526123, 0.05494159460067749,
                   1.3507028967142105)
}
# pdb.set_trace()

# key = [key for key in sections_length.keys() if "capsule_" in key]
CYLINDER_ANYTRAIN = [
    'cylinder_5', 'cylinder_6', 'cylinder_7', 'cylinder_8', 'cylinder_9',
    'cylinder_100'
]
CAPSULE_ANYTRAIN = [
    'capsule_3', 'capsule_4', 'capsule_5', 'capsule_6', 'capsule_7',
    'capsule_8', 'capsule_9', 'capsule_10', 'capsule_11', 'capsule_12',
    'capsule_13'
]
THINCAPSULE2_ANYTRAIN = [
    "thincapsule2_3", "thincapsule2_4", "thincapsule2_5", 'thincapsule2_6',
    'thincapsule2_7', 'thincapsule2_8', 'thincapsule2_9', 'thincapsule2_10'
]
THINCAPSULE3_ANYTRAIN = [
    "thincapsule3_3", "thincapsule3_4", "thincapsule3_5", "thincapsule3_6",
    "thincapsule3_7", 'thincapsule3_8', 'thincapsule3_9', 'thincapsule3_10',
    'thincapsule3_11', 'thincapsule3_12', 'thincapsule3_13', 'thincapsule3_14',
    'thincapsule3_15', 'thincapsule3_16'
]
CONE_ANYTRAIN = ['cone_4', 'cone_5', 'cone_6', 'cone_7', 'cone_8', 'cone_9']
ICOSPHERE_ANYTRAIN = [
    'icosphere_0', 'icosphere_1', 'icosphere_2', 'icosphere_3', 'icosphere_4',
    'icosphere_5', 'icosphere_6', 'icosphere_7', 'icosphere_8', 'icosphere_9'
]
TORUS_ANYTRAIN = [
    'torus_2', 'torus_3', 'torus_4', 'torus_5', 'torus_6', 'torus_7',
    'torus_8', 'torus_9', 'torus_10', 'torus_11'
]
UPTORUS_ANYTRAIN = [
    'uptorus_2', 'uptorus_3', 'uptorus_4', 'uptorus_5', 'uptorus_6',
    'uptorus_7', 'uptorus_8', 'uptorus_9', 'uptorus_10'
]
THINCAPSULE_ANYTRAIN = [
    'thincapsule_2', 'thincapsule_3', 'thincapsule_4', 'thincapsule_5',
    'thincapsule_6', 'thincapsule_7'
]
ANY_TRAIN = [
    'cylinder_3', 'cylinder_4', 'cylinder_5', 'cylinder_6', 'cylinder_7',
    'cylinder_8', 'cylinder_9', 'thincapsule2_6', 'thincapsule2_7',
    'thincapsule2_8', 'thincapsule2_9', 'thincapsule2_10', 'thincapsule_2',
    'thincapsule_3', 'thincapsule_4', 'thincapsule_5', 'thincapsule_6',
    'thincapsule_7', 'thincapsule3_8', 'thincapsule3_9', 'thincapsule3_10',
    'thincapsule3_11', 'thincapsule3_12', 'thincapsule3_13', 'thincapsule3_14',
    'thincapsule3_15', 'thincapsule3_16'
]

# ANY_TRAIN = [
#     'cylinder_3',
#     'cylinder_4',
#     'cylinder_5',
#     'cylinder_6',
#     'cylinder_7',
#     'cylinder_8',
#     'cylinder_9',
# ]
