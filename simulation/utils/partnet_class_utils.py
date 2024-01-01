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
from hand_teleop.utils.bottle4_helper import load_bottle_polygon

BOTTLE_LINK_ORFER1 = ["base", "link_1", "link_0_helper", "link_0", "link_0_0"]
BOTTLE_LINK_ORFER2 = ["base", "link_1", "link_0_helper", "link_0"]
BOTTLE3_SCALE = 0.1
BOTTLE_LIST = [
    "3517", "3520", "3558", "3571", "3574", "3596", "3614", "3615", "3616",
    "3635", "3655", "3678", "3763", "3822", "3854", "3868", "3933", "3934",
    "3990", "4043", "4064", "4084", "4118", "4200", "4216", "4233", "4314",
    "4403", "4427", "4514", "5688", "5861", "6037", "6040", "6222", "6263",
    "6430", "6493", "6771", "6771"
]

# BOTTLE_ANYTRAIN = [
#     "3635", "3520", "3596", "4084", "4118", "4200", "6040", "6222", "6430"
# ]
BOTTLE_ANYTRAIN = [
    [],
]  #1.more than 2cm 2.about 1 to 2 cm,4.different style
# BOTTLE_ANYTRAIN = [
#     [
#         '3635',
#         '4403',
#         '3830',
#         '3868',
#     ],
#     ["4314", "6493", "6037", "4084", "4200", "4118"],
#     ["4216", "6430"],
#     ["4084", "4200", "4118", "4314"],
#     ["4084", "4200", "4118", "6263", "6040", "6493", "6037"],
# ]  #1.more than 2cm 2.about 1 to 2 cm,4.different style
BOTTLE_ANYTRAIN_1 = ["4084", "4200", "4118"]
BOTTLE_ANYTRAIN_2 = ["4084", "4200", "4118"]

KINIFE_ANYTRAIN = [
    "101052", "101057", "101059", "101112", "103582", "103706", "103725"
]  #101062,101085
LIGHTER_ANYTRAIN = [
    "103513", "100348", "103515", "100355", "103503", "100320", "100334",
    "100330", "100334"
]  #100343

DISPENSER_ANYTRAIN = ["101417", "101540", "101463", "103405", "103619"]

BOX_ANYTRAIN = [
    '100174', '100671', '100664', '100685', '47645', '100221', '100243',
    '100189', '102379', '100141', '102456'
]

USB_ANYTRAIN = [
    "100079", "100113", "100511", "101952", "101983", "102042", "102068"
]

OBJECT_ORIENTATION = {
    "bottle": [1, 0, 0, 0],
    "bottle2": [1, 0, 0, 0],
    "bottle3": [0.707, 0.707, 0, 0],
    "bottle4": [1, 0, 0, 0],
    "knife": [0.707, 0, 0, 0.707],
    "dispenser": [0.707, 0, -0, 0.707],
    "lighter": [0.707, 0, 0, 0.707],
    "box": [0, 0, 0, -1],
    "USB": [0.707, 0, 0.707, 0],
    "faucet": [0.707, 0, 0.707, 0],
    "nut": [1, 0, 0, 0]
}

bottle_json = json.load(
    open("./assets/partnet-mobility-dataset/bottle/bottle.json"))
knife_json = json.load(
    open("./assets/partnet-mobility-dataset/knife/knife.json"))
dispenser_json = json.load(
    open("./assets/partnet-mobility-dataset/dispenser/dispenser.json"))
lighter_json = json.load(
    open("./assets/partnet-mobility-dataset/lighter/lighter.json"))
box_json = json.load(open("./assets/partnet-mobility-dataset/box/box.json"))


def calculate_length(points):

    x_min, x_max = np.min(points[:, 0]), np.max(points[:, 0])
    y_min, y_max = np.min(points[:, 1]), np.max(points[:, 1])
    z_min, z_max = np.min(points[:, 2]), np.max(points[:, 2])

    return (x_max - x_min) * BOTTLE3_SCALE, (y_max - y_min) * BOTTLE3_SCALE, (
        z_max - z_min) * BOTTLE3_SCALE, (x_min + x_max) / 2, (
            y_min + y_max) / 2, (z_min + z_max) / 2


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


def save_bbox(dict, link_name, name):

    x_length, y_length, z_length, x_center, y_center, z_center = load_mesh(
        dict['link'][link_name]['geometry'], name)
    dict['link'][link_name]['length'] = [x_length, y_length, z_length]
    dict['link'][link_name]['center'] = [x_center, y_center, z_center]

    return dict


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

    dict = save_bbox(dict, "link_1", name)

    if "link_0" in dict['link'].keys():
        dict = save_bbox(dict, "link_0", name)
    if "link_0_0" in dict['link'].keys():
        dict = save_bbox(dict, "link_0_0", name)

    return dict


def modify_scale(link_dicts, scale):
    x_length_bottle, y_length_bottle, z_length_bottle = link_dicts["link_1"][
        "length"]
    if "link_0_0" in link_dicts.keys():

        x_length_cap, y_length_cap, z_length_cap = link_dicts["link_0_0"][
            "length"]
    else:
        x_length_cap, y_length_cap, z_length_cap = link_dicts["link_0_0"][
            "length"]

    y_scale_cap = (scale[1] / (y_length_cap))[0] * BOTTLE3_SCALE
    x_scale_cap = (scale[0] / (x_length_cap))[0] * BOTTLE3_SCALE
    y_scale_bottle = scale[2] / (y_length_bottle / BOTTLE3_SCALE)

    bottle_scales = [x_scale_cap, y_scale_bottle, x_scale_cap]
    cap_scales = [x_scale_cap, y_scale_cap, x_scale_cap]

    return bottle_scales, cap_scales, y_length_bottle, y_length_cap


def pre_load_mesh(model_name, link_name):
    points = trimesh.load(
        "./assets/partnet-mobility-dataset/bottle/collision/" + model_name +
        "/" + link_name).vertices
    return points


bottle3_align_dict = {}

for any_bottle in BOTTLE_ANYTRAIN:
    for model_name in any_bottle:
        if model_name in bottle3_align_dict.keys():
            continue
        bottle3_align_dict[model_name] = {}
        bottle3_align_dict[model_name]["link_1"] = pre_load_mesh(
            model_name, "link_1.stl")
        bottle3_align_dict[model_name]["link_0_0"] = pre_load_mesh(
            model_name, "link_0_0.stl")


def align_height(model_name, bottle_scales, cap_scales):
    # link1 = trimesh.load(
    #     "./assets/partnet-mobility-dataset/bottle/collision/" + model_name +
    #     "/link_1.stl").vertices
    # link0 = trimesh.load(
    #     "./assets/partnet-mobility-dataset/bottle/collision/" + model_name +
    #     "/link_0_0.stl").vertices
    link1 = bottle3_align_dict[model_name]["link_1"]
    link0 = bottle3_align_dict[model_name]["link_0_0"]
    link1_scale = np.array(link1) * bottle_scales
    link0_scale = np.array(link0) * cap_scales

    dist = np.max(link1_scale[:, 1]) - np.min(link0_scale[:, 1])

    return [0, dist, 0]


def build_articulation(dict,
                       scene,
                       model_name,
                       scale=None,
                       random=True,
                       visual_only=False):

    builder: sapien.ArticulationBuilder = scene.create_articulation_builder()
    if visual_only:
        builder = scene.create_actor_builder()

    link_dicts = dict["link"]
    links = {}
    # random = True

    # if random:
    bottle_scales, cap_scales, y_length_bottle, y_length_cap = modify_scale(
        link_dicts, scale)

    # else:
    #     bottle_scales, cap_scales = [
    #         BOTTLE3_SCALE, BOTTLE3_SCALE, BOTTLE3_SCALE
    #     ], [BOTTLE3_SCALE, BOTTLE3_SCALE, BOTTLE3_SCALE]

    # if "link_0_0" in link_dicts.keys():
    #     links_names = BOTTLE_LINK_ORFER1
    # else:
    #     links_names = BOTTLE_LINK_ORFER2
    links_names = BOTTLE_LINK_ORFER1

    for name in links_names:

        if not visual_only:

            link = builder.create_link_builder()
            link.set_name(name)

        if name in ["link_1"]:
            scales = bottle_scales
            xyz = [0, 0, 0]
        else:
            xyz = [0, 0, 0.0]

            scales = cap_scales

        if bool(link_dicts[name]):

            if name in ["link_1"]:

                for mesh_file in link_dicts[name]["geometry"]:

                    if name in ["link_1"]:
                        scales = bottle_scales
                    else:
                        scales = cap_scales
                    if not visual_only:

                        link.add_multiple_collisions_from_file(
                            mesh_file,
                            pose=sapien.Pose([0, 0, 0], [1, 0, 0, 0]),
                            scale=scales,
                            density=1000)

                        link.add_visual_from_file(mesh_file,
                                                  pose=sapien.Pose(
                                                      [0, 0, 0], [1, 0, 0, 0]),
                                                  scale=scales)

                    else:
                        builder.add_visual_from_file(
                            mesh_file,
                            pose=sapien.Pose([0, 0, 0], [1, 0, 0, 0]),
                            scale=scales)

            else:

                if not visual_only:

                    link.add_multiple_collisions_from_file(
                        "./assets/partnet-mobility-dataset/bottle/collision/" +
                        model_name + "/" + name + ".stl",
                        pose=sapien.Pose([0, 0, 0], [1, 0, 0, 0]),
                        scale=scales,
                        density=1000)

                    link.add_visual_from_file(
                        "./assets/partnet-mobility-dataset/bottle/collision/" +
                        model_name + "/" + name + ".stl",
                        pose=sapien.Pose([0, 0, 0], [1, 0, 0, 0]),
                        scale=scales)

                else:
                    builder.add_visual_from_file(
                        "./assets/partnet-mobility-dataset/bottle/collision/" +
                        model_name + "/" + name + ".stl",
                        pose=sapien.Pose([0, 0, 0], [1, 0, 0, 0]),
                        scale=scales)

        if not visual_only:
            links[name] = link

    joint_dicts = dict["joint"]
    joints = {}

    if not visual_only:

        continuous_joint_info = {}

        for name in joint_dicts.keys():

            child_name = joint_dicts[name]['child']
            parent_name = joint_dicts[name]['parent']

            if joint_dicts[name]['type'] == "fixed":
                xyz = (np.array(
                    joint_dicts[name]['origin']['xyz'].split()).astype(
                        np.float32))

                links[child_name].set_joint_properties(
                    joint_type="fixed",
                    limits=[],
                    pose_in_parent=sapien.Pose([0, 0, 0], [1, 0, 0, 0]))

            if joint_dicts[name]['type'] == "prismatic":

                limit = joint_dicts[name]['limit']

                xyz = (np.array(
                    joint_dicts[name]['origin']['xyz'].split()).astype(
                        np.float32))[[0, 2, 1]] * 0

                links[continuous_joint_info["child_name"].get_name(
                )].set_joint_properties(joint_type="revolute",
                                        limits=[[-2 * np.pi, 2 * np.pi]],
                                        pose_in_child=sapien.Pose([0, 0, 0], [
                                            0.707,
                                            0,
                                            0,
                                            0.707,
                                        ]),
                                        pose_in_parent=sapien.Pose(
                                            [0, 0, 0], [0.707, 0, 0, 0.707]))

                xyz = align_height(model_name, bottle_scales, cap_scales)

                links[child_name].set_joint_properties(
                    joint_type="prismatic",
                    limits=[[-0.00, 0.00]],
                    pose_in_child=sapien.Pose([0, 0, 0], [0.707, 0, 0, 0.707]),
                    pose_in_parent=sapien.Pose(xyz, [0.707, 0, 0, 0.707]))

            if joint_dicts[name]['type'] == "continuous":
                continuous_joint_info["joint_type"] = "revolute"
                continuous_joint_info["child_name"] = links[child_name]
                continuous_joint_info["limits"] = [[-2 * np.pi, 2 * np.pi]]
                continuous_joint_info["xyz"] = np.array(
                    joint_dicts[name]['origin']['xyz'].split()).astype(
                        np.float32)[[0, 2, 1]]
                # pdb.set_trace()
                # continuous_joint_info["pose_in_child"] = sapien.Pose(
                #     [0, 0, 0], [0.707, 0, 0.707, 0])
                # continuous_joint_info["pose_in_parent"] = sapien.Pose(
                #     [0, 0, 0], [0.707, 0, 0.707, 0])

            links[child_name].set_parent(links[parent_name].get_index())
        for link in builder.get_link_builders():
            link.set_collision_groups(1, 1, 4, 4)

    if not visual_only:

        object = builder.build(fix_root_link=True)
    else:
        object = builder.build_static()
    render_scale = np.array(cap_scales)  #[[0, 2, 1]]

    return object, render_scale


def get_partnet_root_dir():
    current_dir = Path(__file__).parent
    partnet_dir = current_dir.parent.parent / "assets" / "partnet-mobility-dataset"
    return partnet_dir.resolve()


def build_static(scene, model_type, model_name, scale):
    builder = scene.create_actor_builder()

    visual_file = "./assets/partnet-mobility-dataset/" + model_type + "/visual/" + model_name + "/" + model_name + ".stl"

    if model_name in ["3763"]:
        r = R.from_rotvec(np.pi / 2 * np.array([1, 1, 0]))
        pose = r.as_quat()[[3, 0, 1, 2]]
    else:
        pose = [0.7071, 0.7071, 0, 0]

    if model_type in ["lighter"]:
        r = R.from_euler('xyz', [-90, 0, 90], degrees=True)
        pose = r.as_quat()[[3, 0, 1, 2]]  #x,y,z,w
        # pose = [1, 0, 0, 0]

    builder.add_visual_from_file(filename=visual_file,
                                 scale=np.array([scale, scale, scale]),
                                 pose=sapien.Pose(q=np.array(pose)))

    actor = builder.build_static(name=model_name)

    return actor


bottle3_dict = {}

for any_bottle in BOTTLE_ANYTRAIN:
    for model_name in any_bottle:
        if model_name in bottle3_dict.keys():
            continue
        bottle3_dict[model_name] = {}
        bottle3_dict[model_name] = parse_urdf(
            "../sapien_task/assets/partnet-mobility-dataset/bottle/" +
            model_name + "/mobility.urdf", bottle3_dict[model_name],
            model_name)


def load_partnet_object(scene: sapien.Scene,
                        model_type: str,
                        model_name: str,
                        physical_material: sapien.PhysicalMaterial = None,
                        density=1000,
                        visual_only=False,
                        random=False,
                        scale=None,
                        renderer=None,
                        down=False):

    if model_type.lower() == "bottle3":

        object, render_scale = build_articulation(bottle3_dict[model_name],
                                                  scene, model_name, scale,
                                                  random, visual_only)
        height = 0.2

        return object, height, render_scale
    if model_type.lower() == "bottle4":

        object, render_scale = load_bottle_polygon(scene, model_name, scale,
                                                   renderer, down)
        height = 0.2

        return object, height, render_scale
    filename = "./assets/partnet-mobility-dataset" + "/" + model_type + "/" + model_name + "/" + "mobility.urdf"

    loader = scene.create_urdf_loader()

    if model_type.lower() == "bottle":
        loader.scale = 0.06

        # if random:

        #     loader.scale = np.random.uniform(0.08, 0.12, 1)
        #     if model_name in ["3520", "3635", "3596"]:
        #         loader.scale = np.random.uniform(0.1, 0.15, 1)
        #     if model_name in ["3763", "3616", "4043"]:
        #         loader.scale = np.random.uniform(0.15, 0.20, 1)  # need bigger
        #     if model_name in ["4064"]:
        #         loader.scale = np.random.uniform(0.2, 0.3, 1)  # need bigger
        #     if model_name in ["3520", "8736"]:
        #         loader.scale = np.random.uniform(0.10, 0.15, 1)
        #     if model_name in ["3868"]:
        #         loader.scale = np.random.uniform(0.12, 0.18, 1)
        #     if model_name in ["3830"]:
        #         loader.scale = np.random.uniform(0.12, 0.15, 1)

        # else:
        #     loader.scale = 0.08
        # if model_name in ["3655", "3615", "3616", "4043"]:

        #     loader.scale = 0.15

        # if model_name in ["4064"]:

        #     loader.scale = 0.2
        height = 0.2

    if model_type.lower() == "bottle2":
        if random:

            loader.scale = 0.06

            # if model_name in ["3520", "3635", "3596"]:
            #     loader.scale = np.random.uniform(0.1, 0.15, 1)
            # if model_name in ["3763", "3616", "4043"]:
            #     loader.scale = np.random.uniform(0.15, 0.20, 1)  # need bigger
            # if model_name in ["4064"]:
            #     loader.scale = np.random.uniform(0.2, 0.3, 1)  # need bigger
            # if model_name in ["3520", "8736"]:
            #     loader.scale = np.random.uniform(0.10, 0.15, 1)
            # if model_name in ["3868"]:
            #     loader.scale = np.random.uniform(0.12, 0.18, 1)
            # if model_name in ["3830"]:
            #     loader.scale = np.random.uniform(0.12, 0.15, 1)

        else:
            loader.scale = 0.1
            if model_name in ["3655", "3615", "3616", "4043"]:

                loader.scale = 0.15

            if model_name in ["4064"]:

                loader.scale = 0.2
        height = 0.05

    if model_type.lower() == "faucet":
        object, render_scale = build_articulation_faucet(scene, model_name)
        height = 0.2

        return object, height, [loader.scale, loader.scale, loader.scale]
        # loader.scale = 0.2
        # height = 0.15
        # if random:

        #     loader.scale = np.random.uniform(0.15, 0.2, 1)

    if model_type.lower() == "nut":
        height = 0.0
        loader.scale = 18

    if model_type.lower() == "lighter":
        height = 0.1
        loader.scale = 0.2

    material = scene.create_physical_material(1, 1, 0)

    material.set_static_friction(10)
    material.set_dynamic_friction(10)

    config = {'material': material, 'density': 1000}
    builder = loader.load_file_as_articulation_builder(filename, config)

    for link in builder.get_link_builders():
        link.set_collision_groups(1, 1, 4, 4)

    if not visual_only:
        loader.load_multiple_collisions_from_file = True

        object = builder.build(fix_root_link=True)
    else:

        if model_name not in ["3763"]:
            object = build_static(scene, model_type, model_name, loader.scale)
        else:
            return None, height, loader.scale

    return object, height, [loader.scale, loader.scale, loader.scale]


def load_free_partnet_object(scene: sapien.Scene,
                             object_name,
                             scale=0.07,
                             visual_only=False,
                             material=None,
                             static=False):

    filename = "./assets/partnet-mobility-dataset/bottle4/" + (
        object_name) + ".stl"
    builder = scene.create_actor_builder()
    scales = np.array([scale] * 3)
    density = 1000
    if material is None:
        material = scene.engine.create_physical_material(100000000, 0.5, 1)

    if not visual_only:
        builder.add_multiple_collisions_from_file(str(filename),
                                                  scale=scales,
                                                  density=density,
                                                  material=material)

    builder.add_visual_from_file(str(filename), scale=scales)
    if not visual_only and not static:
        actor = builder.build()
    else:
        actor = builder.build_static()
    return actor


with open(
        'assets/partnet-mobility-dataset/drawer/info_cabinet_drawer_train.json',
        'r') as f:
    drawer_info = json.load(f)


def load_drawer_object(
    scene: sapien.Scene,
    model_type: str,
    model_name: str,
    physical_material: sapien.PhysicalMaterial = None,
    density=1000,
    visual_only=False,
    renderer=None,
):

    filename = "./assets/partnet-mobility-dataset" + "/" + model_type + "/" + model_name + "/" + "mobility.urdf"

    loader = scene.create_urdf_loader()
    loader.scale = drawer_info[model_name]["scale"] / 2.0

    material = scene.create_physical_material(1, 1, 0)

    material.set_static_friction(10)
    material.set_dynamic_friction(10)

    config = {'material': material, 'density': 1000}
    builder = loader.load_file_as_articulation_builder(filename, config)

    for link in builder.get_link_builders():
        link.set_collision_groups(1, 1, 4, 4)

    object = builder.build(fix_root_link=True)

    height = 0.1

    return object, height, [loader.scale, loader.scale, loader.scale]