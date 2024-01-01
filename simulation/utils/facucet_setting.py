import numpy as np
import trimesh
import pdb
import xml.etree.ElementTree as ET
import sapien.core as sapien
from scipy.spatial.transform import Rotation as R

FAUCET_LIST = ["101374", "912", "102090", "102100", "101421", "1823", "101397"]
FAUCRT_LINK_ORFER = ["base", "link_1", "link_0"]
FAUCET_ROTATION = {
    "101374": [1, 0, 0, 0],
    "912": [1, 0, 0, 0],
    "102090": [0.707, 0, 0.707, 0],
    "102100": [0.707, 0, 0.707, 0],
    "101421": [0, 0, -1, 0],
    "1823": [1, 0, 0, 0],
    "101397": [0.707, 0, 0.707, 0]
}

FAUCET_HAND_HEIGHT = {
    "101374": -0.05,
    "912": 0.1,
    "102090": -0.03,
    "102100": 0.0,
    "101421": -0.05,
    "1823": -0.05,
    "101397": -0.05
}
FAUCET_SCALE = {
    "101374": 0.2,
    "912": 0.2,
    "102090": 0.2,
    "102100": 0.2,
    "101421": 0.2,
    "1823": 0.12,
    "101397": 0.2
}

FAUCET_SINGLE = ["102090", "1823", "101397", "101374"]

RADIUS = 0.25
HEIGHT = 0.1


def load_mesh(name):
    # for name in mesh_list:
    mesh = trimesh.load("./assets/partnet-mobility-dataset/faucet/collision/" +
                        name + "/link_0.stl")
    points = mesh.vertices
    faces = mesh.faces

    x_min, y_min, z_min = np.min(points, axis=0)
    x_max, y_max, z_max = np.max(points, axis=0)
    x_length, y_length, z_length = x_max - x_min, y_max - y_min, z_max - z_min
    heigth_index = np.argmin([x_length, y_length, z_length])

    max_radius_index = np.argmax([x_length, y_length, z_length])
    length = [x_length, y_length, z_length]
    # print("===========================")
    # print(name)
    # print("min", x_min, y_min, z_min)
    # print("max", x_max, y_max, z_max)
    # print("length", x_length, y_length, z_length)
    # print("height index", heigth_index)
    scale = np.ones(3) * RADIUS / length[max_radius_index]
    scale[heigth_index] = HEIGHT / length[heigth_index]
    # print(scale)
    return scale, 0.2


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
                                    "./assets/partnet-mobility-dataset/faucet/"
                                    + name + "/" + ccc.attrib["filename"])

    # dict = save_bbox(dict, "link_1", name)

    # if "link_0" in dict['link'].keys():
    #     dict = save_bbox(dict, "link_0", name)
    # if "link_0_0" in dict['link'].keys():
    #     dict = save_bbox(dict, "link_0_0", name)

    return dict


FAUCET_DICT = {}
FAUCET_URDF_DICT = {}
for name in FAUCET_LIST:

    scale, height = load_mesh(name)
    FAUCET_DICT[name] = {}
    FAUCET_DICT[name]["scale"] = {}
    FAUCET_DICT[name]["height"] = {}
    FAUCET_DICT[name]["scale"] = scale
    FAUCET_DICT[name]["height"] = height
    FAUCET_URDF_DICT[name] = {}
    FAUCET_URDF_DICT[name] = parse_urdf(
        "../sapien_task/assets/partnet-mobility-dataset/faucet/" + name +
        "/mobility.urdf", FAUCET_URDF_DICT[name], name)


def load_faucet(name):
    return FAUCET_DICT[name]['scale'], FAUCET_DICT[name]['height']


def build_articulation_faucet(scene, model_name):

    builder: sapien.ArticulationBuilder = scene.create_articulation_builder()

    link_dicts = FAUCET_URDF_DICT[model_name]["link"]
    links = {}

    scales = FAUCET_DICT[model_name]["scale"]

    links_names = FAUCRT_LINK_ORFER

    for name in links_names:
        link = builder.create_link_builder()
        link.set_name(name)

        if bool(link_dicts[name]):

            if name in ["link_1"]:

                for mesh_file in link_dicts[name]["geometry"]:

                    link.add_multiple_collisions_from_file(
                        mesh_file,
                        pose=sapien.Pose([0, 0, 0], [1, 0, 0, 0]),
                        scale=scales)
                    # if (name not in ['link_0'] and "link_0_0" in links_names) or (
                    #         name in ['link_0'] and "link_0_0" not in links_names):

                    link.add_visual_from_file(mesh_file,
                                              pose=sapien.Pose([0, 0, 0],
                                                               [1, 0, 0, 0]),
                                              scale=scales)

            else:

                link.add_multiple_collisions_from_file(
                    "./assets/partnet-mobility-dataset/faucet/collision/" +
                    model_name + "/" + name + ".stl",
                    pose=sapien.Pose([0, 0, 0], [1, 0, 0, 0]),
                    scale=scales)
                link.add_visual_from_file(
                    "./assets/partnet-mobility-dataset/faucet/collision/" +
                    model_name + "/" + name + ".stl",
                    pose=sapien.Pose([0, 0, 0], [1, 0, 0, 0]),
                    scale=scales)

        links[name] = link

    joint_dicts = FAUCET_URDF_DICT[model_name]["joint"]

    continuous_joint_info = {}
    for name in joint_dicts.keys():

        child_name = joint_dicts[name]['child']
        parent_name = joint_dicts[name]['parent']

        if joint_dicts[name]['type'] == "fixed":
            xyz = (np.array(joint_dicts[name]['origin']['xyz'].split()).astype(
                np.float32))
            rpy = (np.array(joint_dicts[name]['origin']['rpy']).astype(
                np.float32))
            r = R.from_rotvec(rpy)
            quat = r.as_quat()[[3, 0, 1, 2]]

            links[child_name].set_joint_properties(joint_type="fixed",
                                                   limits=[],
                                                   pose_in_parent=sapien.Pose(
                                                       xyz, quat))

        if joint_dicts[name]['type'] == "continuous":

            xyz = (np.array(joint_dicts[name]['origin']['xyz'].split()).astype(
                np.float32))
            # rpy = (np.array(joint_dicts[name]['origin']['rpy']).astype(
            #     np.float32))

            # r = R.from_rotvec(rpy)
            # quat = r.as_quat()[[3, 0, 1, 2]]
            quat = [1, 0, 0, 0]
            continuous_joint_info["joint_type"] = "revolute"
            continuous_joint_info["child_name"] = links[child_name]
            continuous_joint_info["limits"] = [[-2 * np.pi, 2 * np.pi]]
            continuous_joint_info["xyz"] = np.array(
                joint_dicts[name]['origin']['xyz'].split()).astype(
                    np.float32)[[0, 2, 1]]
            links[continuous_joint_info["child_name"].get_name(
            )].set_joint_properties(joint_type="revolute",
                                    limits=[[-2 * np.pi, 2 * np.pi]],
                                    pose_in_child=sapien.Pose(xyz, quat),
                                    pose_in_parent=sapien.Pose(xyz, quat))

        links[child_name].set_parent(links[parent_name].get_index())
    for link in builder.get_link_builders():
        link.set_collision_groups(1, 1, 4, 4)
    object = builder.build(fix_root_link=True)
    render_scale = np.array(scales)  #[[0, 2, 1]]

    return object, render_scale
