from http.client import PARTIAL_CONTENT
import json
from pathlib import Path

import numpy as np
import sapien.core as sapien

import pdb

# PEN_ANYTRAIN = [
#     '101736',
#     '101713',
#     '101712',
#     '101732',
#     '101703',
#     '101727',
#     '101698',
# ]
PEN_ANYTRAIN = ["101787", "101732", "101727", "102922", "102973"]
PEN_ANYTRAIN2 = ["101787", "101732", "101727"]  #100085
PEN_ANYTRAIN_NOVEL = [
    "101698", "101703", "101712", "101713", "101714", "101736", "102922",
    "102973"
]
USB_ANYTRAIN = [
    '100085', '100095', '100133', '100072', '100108', '100128', '100113',
    '100123', '100087', '100513', '101983', '100511'
]
USB_ANYTRAIN_NOVEL = [
    "100073", "100079", "100103", "101952", "102062", "102024", "102021"
]
ANY_KIND = ["pen", "spoon"]
ALL_KIND = ["pen", "box", "bottle"]

KNIFE_ANYTRAIN = [
    '101052', '101057', '101059', '101107', "101660", "102400", "103572",
    "103716", "103733"
]
PARTNET_ORIENTATION = {
    "101540": [0, 0, 0, 1],
    "100335": [1, 0, 0, 0],
    "3520": [1, 0, 0, 0],
    "100885": [0.707, 0, 0.707, 0],
    "3635": [1, 0, 0, 0],
    "3398": [1, 0, 0, 0],
    "4084": [1, 0, 0, 0],
    "6263": [1, 0, 0, 0],
    "101336": [1, 0, 0, 0],
    "101531": [1, 0, 0, 0],
    "101052": [0.707, 0, 0.707, 0],
    "101786": [0, 0, -1, 0],
    "100285": [1, 0, 0, 0],
    "101062": [0.707, 0, 0.707, 0],
    "102290": [1, 0, 0, 0],
    "103416": [1, 0, 0, 0],
    "100348": [1, 0, 0, 0],
    "101563": [0, 0, 0, 1],
    "101112": [0.707, 0, 0.707, 0],
    "101057": [0.707, 0, 0.707, 0],
    "101528": [0, 0, 0, 1],
    "102872": [0.707, 0, 0.707, 0],
    "100955": [0.707, 0, 0.707, 0],
    "102181": [0, 0, 0, 1],
    "100330": [0.707, 0, 0, 0.707],  #lighter
    "102940": [1, 0, 0, 0],  #pen,
    "100247": [0.707, 0, 0, -0.707],  #box
    "101489": [1, 0, 0, 0],  # press ,
    "101417": [0, 0, 0, 1],  # press ,dispenser
    "101463": [0, 0, 0, 1],
}

PARTNET_SCALE = {
    "101540": 0.2,
    "100335": 0.15,
    "3520": 0.06,
    "100885": 0.1,
    "3635": 0.1,
    "3398": 0.1,
    "4084": 0.09,
    "6263": 0.1,
    "101336": 0.1,
    "101531": 0.1,
    "101052": 0.15,
    "101786": 0.1,
    "100285": 0.08,
    "101062": 0.1,
    "102290": 0.1,
    "103416": 0.15,
    "100348": 0.1,
    "101563": 0.1,
    "101112": 0.1,
    "101057": 0.1,
    "101528": 0.2,
    "102872": 0.1,
    "100955": 0.1,
    "102181": 0.1,
    "100330": 0.12,
    "102940": 0.2,
    "100247": 0.1,
    "101489": 0.1,
    "101417": 0.3,
    "101463": 0.2
}
PARTNET_HEIGHT = {
    "101540": 0.2,
    "100335": 0.10,
    "3520": 0.06,
    "100885": 0.05,
    "3635": 0.1,
    "3398": 0.05,
    "4084": 0.1,
    "6263": 0.1,
    "101336": 0.1,
    "101531": 0.1,
    "101052": 0.2,
    "101786": 0.1,
    "100285": 0.1,
    "101062": 0.1,
    "102290": 0.1,
    "103416": 0.1,
    "100348": 0.1,
    "101563": 0.1,
    "101112": 0.1,
    "101057": 0.1,
    "101528": 0.09,
    "102872": 0.1,
    "100955": 0.1,
    "102181": 0.1,
    "100330": 0.12,
    "102940": 0.3,
    "100247": 0.04,
    "101489": 0.05,
    "101417": 0.15,
    "101463": 0.1
}

PARTNET_HAND_POSE = {"101052": [[0.137, 0.02, 0.32], [0.707, 0, 0, 0.707]]}

PARTNET_ADJUSTMENT = {"100885": [0.45, 0, 0], "101052": [0.0, 0, 0]}
PARTNET_LIMIT = {
    "4084": [[-0.006, 0.024]],
    "3635": [[-0.018, 0.008]],
    "100247": [[-0.576, 2.513]]
}

BOTTLE_ANYTRAIN = ["4118", "3380", "3520", "6771", "5902", "3990"]
BUCKET_ANYTRAIN = [
    "4001", "100442", "100431", "100432", "100438", "100454", "102358"
]

import xml.etree.ElementTree as ET

BREAKING_ANYTRAIN = [
    "buser_head", "druid", "alien_pilot_tiny", "archer", "royal_wizard",
    "luckysquid01"
]

BREAKING_BOTTLE_ANYTRAIN = ["WineGlass8-mode_14_8pcs"]
BOX_ANYTRAIN = ["100295", "100334", "100340"]  #, "100313","100350"


def parse_urdf(file, dict, name, model_cat):

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

                # if c.tag in ["collision"]:
                #     continue

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

                            if "collision" not in dict['link'][
                                    child.attrib['name']].keys():
                                dict['link'][
                                    child.attrib['name']]['collision'] = []
                            if "visual" not in dict['link'][
                                    child.attrib['name']].keys():
                                dict['link'][
                                    child.attrib['name']]['visual'] = []

                                # print(child.tag, child.attrib["name"])
                                # print(c.tag, ccc.attrib["filename"])

                            dict['link'][child.attrib['name']][c.tag].append(
                                "./assets/partnet-mobility-dataset/" +
                                model_cat + "/" + name + "/" +
                                ccc.attrib["filename"])
    return dict


def get_partnet_root_dir():
    current_dir = Path(__file__).parent
    partnet_dir = current_dir.parent.parent / "assets" / "partnet-mobility-dataset"
    # info_path = modelnet_dir / "info.json"
    # with info_path.open("r") as f:
    #     cat_scale = json.load(f)

    return partnet_dir.resolve()


def load_partnet_object(scene: sapien.Scene,
                        model_cat: str,
                        model_name: str,
                        physical_material: sapien.PhysicalMaterial = None,
                        density=1000,
                        visual_only=False):

    filename = "./assets/partnet-mobility-dataset" + "/" + model_cat + "/" + model_name + "/" + "mobility.urdf"
    loader = scene.create_urdf_loader()
    loader.scale = 0.1
    builder = loader.load_file_as_articulation_builder(filename)

    loader.load_multiple_collisions_from_file = True
    if not visual_only:
        object = builder.build(fix_root_link=False)

    return object, 0.1, loader.scale


def load_partnet_actor(scene: sapien.Scene,
                       model_cat: str,
                       model_name: str,
                       physical_material: sapien.PhysicalMaterial = None,
                       density=1000,
                       visual_only=False,
                       height=None):

    r = R.from_euler('xyz', [90, 0, 90], degrees=True)
    orientation = r.as_quat()[[3, 0, 1, 2]]

    r = R.from_euler('xyz', [90, 0, 90], degrees=True)
    bucket_orientation = r.as_quat()[[3, 0, 1, 2]]

    cat_orientation = {
        "pen": orientation,
        "USB": [0.707, 0, 0, 0.707],
        "bottle": [0.707, 0.707, 0, 0],
        "box": bucket_orientation,
        "bucket": bucket_orientation
    }
    builder = scene.create_actor_builder()
    if model_cat in ["bottle1", "box"]:
        files = os.listdir("./assets/partnet-mobility-dataset" + "/" +
                           model_cat + "/" + model_name + "/" + "new_objs")
        visual_files = os.listdir("./assets/partnet-mobility-dataset" + "/" +
                                  model_cat + "/" + model_name + "/" +
                                  "textured_objs")

    else:
        files = os.listdir("./assets/partnet-mobility-dataset" + "/" +
                           model_cat + "/" + model_name + "/" +
                           "textured_objs")

    collision_files = []
    for f in files:
        if model_cat in ["bottle1", "box"]:
            if not f.endswith("obj") and not f.endswith(
                    "stl") or "convex" in f:
                continue
            collision_files.append("./assets/partnet-mobility-dataset" + "/" +
                                   model_cat + "/" + model_name + "/" +
                                   "new_objs/" + f)

        else:
            collision_files.append("./assets/partnet-mobility-dataset" + "/" +
                                   model_cat + "/" + model_name + "/" +
                                   "textured_objs/" + f)
    visual_tex_files = []
    if model_cat in ["bottle1", "box"]:

        for f in visual_files:
            visual_tex_files.append("./assets/partnet-mobility-dataset" + "/" +
                                    model_cat + "/" + model_name + "/" +
                                    "textured_objs/" + f)

    x_length, y_length, z_length, x_center, y_center, z_center = load_mesh(
        collision_files, quat=np.array(cat_orientation[model_cat]))
    material = scene.create_physical_material(100, 100, 0)
    scales = np.array([0.02547181 / y_length] * 3)
    if height is not None:
        scales = np.array([
            0.02547181 / y_length,
            height / z_length,
            0.02547181 / y_length,
        ])

    if model_cat in ["bucket"]:
        scales = np.array([
            0.05547181 / y_length,
            height / z_length / 1.2,
            0.05547181 / y_length,
        ])
    elif model_cat in ["box"]:
        scales = np.array([
            0.03547181 / y_length,
            height / z_length,
            0.05547181 / x_length,
        ])

    for filename in collision_files:
        if not filename.endswith("obj") and not filename.endswith(
                "stl") or "convex" in filename:
            continue

        density = 1000
        if model_cat in ["bottle1", "bucket"]:
            density = 3000
        elif model_cat in ["box"]:
            density = 500

        if not visual_only:
            builder.add_multiple_collisions_from_file(
                filename=filename,
                scale=scales,
                density=density,
                # material=material,
                pose=sapien.Pose(q=cat_orientation[model_cat]))

    if model_cat in ["bottle1", "box"]:
        collision_files = visual_tex_files

    for filename in collision_files:

        if not filename.endswith("obj") and not filename.endswith(
                "stl") or "convex" in filename:
            continue

        builder.add_visual_from_file(
            filename=filename,
            scale=scales,
            pose=sapien.Pose(q=cat_orientation[model_cat]))

    if not visual_only:
        actor = builder.build()
    else:
        actor = builder.build_static(name=model_name)

    return actor, np.array(
        scales[[0, 2,
                1]]), abs(z_center - abs(z_length / 2)) * scales[1], np.array(
                    [x_length, y_length, z_length]) * scales[[0, 2, 1]]


def load_breaking_bed(scene: sapien.Scene,
                      model_cat: str,
                      model_name: str,
                      physical_material: sapien.PhysicalMaterial = None,
                      density=1000,
                      visual_only=False,
                      height=None):

    r = R.from_euler('xyz', [0, 0, 90], degrees=True)
    orientation = r.as_quat()[[3, 0, 1, 2]]
    if model_name in BREAKING_BOTTLE_ANYTRAIN:
        r = R.from_euler('xyz', [90, 0, 90], degrees=True)
        orientation = r.as_quat()[[3, 0, 1, 2]]

    builder = scene.create_actor_builder()

    collision_files = []
    collision_files.append(
        "./assets/partnet-mobility-dataset/breakingbed/convex_mesh/" +
        model_name + '.stl')

    x_length, y_length, z_length, x_center, y_center, z_center = load_mesh(
        collision_files, quat=np.array(orientation))
    material = scene.create_physical_material(100, 100, 0)
    scales = np.array([0.02547181 / y_length] * 3)
    if height is not None:
        scales = np.array([
            0.06547181 / y_length,
            0.06547181 / y_length,
            0.0647181 / y_length,
        ])
    if model_name in BREAKING_BOTTLE_ANYTRAIN:
        scales = np.array([
            height / z_length,
            height / z_length,
            height / z_length,
        ])

    density = 1000

    if not visual_only:
        builder.add_multiple_collisions_from_file(
            filename=
            "./assets/partnet-mobility-dataset/breakingbed/convex_mesh/" +
            model_name + '.stl',
            scale=scales,
            density=density,
            # material=material,
            pose=sapien.Pose(q=orientation))

    builder.add_visual_from_file(
        filename="./assets/partnet-mobility-dataset/breakingbed/visual_mesh/" +
        model_name + '.stl',
        scale=scales,
        pose=sapien.Pose(q=orientation))

    if not visual_only:
        actor = builder.build()
    else:
        actor = builder.build_static(name=model_name)

    return actor, np.array(
        scales), abs(z_center - abs(z_length / 2)) * scales[1], np.array(
            [x_length, y_length, z_length]) * scales[[0, 2, 1]]


# MUG_ANYTRAIN = [
#     "1c3fccb84f1eeb97a3d0a41d6c77ec7c", "1f035aa5fc6da0983ecac81e09b15ea9",
#     "5c48d471200d2bf16e8a121e6886e18d", "5fe74baba21bba7ca4eec1b19b3a18f8",
#     "7a8ea24474846c5c2f23d8349a133d2b", "8b780e521c906eaf95a4f7ae0be930ac",
#     "71ca4fc9c8c29fa8d5abaf84513415a2", "85a2511c375b5b32f72755048bac3f96",
#     "403fb4eb4fc6235adf0c7dbe7f8f4c8e", "1305b9266d38eb4d9f818dd0aa1a251",
#     "2997f21fa426e18a6ab1a25d0e8f3590", "3143a4accdc23349cac584186c95ce9b",
#     "9196f53a0d4be2806ffeedd41ba624d6", "9737c77d3263062b8ca7a0a01bcd55b6",
#     "9961ccbafd59cb03fe36eb2ab5fa00e0", "34869e23f9fdee027528ae0782b54aae",
#     "162201dfe14b73f0281365259d1cf342", "187859d3c3a2fd23f54e1b6f41fdd78a",
#      "a8f7a0edd3edc3299e54b4084dc33544"
# ]

MUG_ANYTRAIN = [
    "1c3fccb84f1eeb97a3d0a41d6c77ec7c", "5c48d471200d2bf16e8a121e6886e18d",
    "8b780e521c906eaf95a4f7ae0be930ac", "71ca4fc9c8c29fa8d5abaf84513415a2",
    "159e56c18906830278d8f8c02c47cde0"
]

# 5d72df6bc7e93e6dd0cd466c08863ebd 35ce7ede92198be2b759f7fb0032e59 68f4428c0b38ae0e2469963e6d044dfe 159e56c18906830278d8f8c02c47cde0
# 639a1f7d09d23ea37d70172a29ade99a 1a97f3c83016abca21d0de04f408950f
MUG_ANYTRAIN = ["9961ccbafd59cb03fe36eb2ab5fa00e0"]
import pickle
with open("./assets/shapenet/03797390/mug.pkl", "rb") as file:
    MUG_ANYTRAIN_dict = pickle.load(file)


def load_shapenet(scene: sapien.Scene,
                  model_cat: str,
                  model_name: str,
                  physical_material: sapien.PhysicalMaterial = None,
                  density=1000,
                  visual_only=False,
                  height=None):

    # orientation = [1, 0, 0, 0]
    r = R.from_euler('xyz', [90, 0, 180], degrees=True)
    orientation = r.as_quat()[[3, 0, 1, 2]]

    builder = scene.create_actor_builder()

    files = os.listdir("./assets/shapenet/" + model_cat + "/" + model_name +
                       "/collision/")

    collision_files = []
    for f in files:
        collision_files.append("./assets/shapenet/" + model_cat + "/" +
                               model_name + "/collision/" + f)

    x_length, y_length, z_length, x_center, y_center, z_center = load_mesh(
        collision_files, quat=np.array(orientation))
    material = scene.create_physical_material(100, 100, 0)

    x_handle_length, y_handle_length, z_handle_length, _, _, _ = load_mesh(
        collision_files[-1], quat=np.array(orientation))

    scales = np.array([0.08 / y_length] * 3)
    # if height is not None:
    #     scales = np.array([
    #         0.2547181 / y_length,
    #         height / z_length,
    #         0.2547181 / y_length,
    #     ])

    position = [-MUG_ANYTRAIN_dict[model_name] * scales[0], 0, 0]

    for filename in collision_files:
        if not filename.endswith("obj"):
            continue

        if not visual_only:
            builder.add_multiple_collisions_from_file(
                filename=filename,
                scale=scales,
                density=30,
                # material=material,
                pose=sapien.Pose(q=orientation, p=position))

    builder.add_visual_from_file(filename="./assets/shapenet/" + model_cat +
                                 "/" + model_name + "/model.obj",
                                 scale=scales,
                                 pose=sapien.Pose(q=orientation, p=position))

    if not visual_only:
        actor = builder.build()
    else:
        actor = builder.build_static(name=model_name)

    return actor, np.array(
        scales[[0, 2,
                1]]), abs(z_center - abs(z_length / 2)) * scales[1], np.array(
                    [x_length, y_length, z_length]) * scales[[0, 2, 1]]


def load_knife_actor(scene: sapien.Scene,
                     model_cat: str,
                     model_name: str,
                     physical_material: sapien.PhysicalMaterial = None,
                     density=1000,
                     visual_only=False,
                     height=None):

    cat_orientation = {"knife": [0, 1, 0, 0]}
    builder = scene.create_actor_builder()

    files = os.listdir("./assets/partnet-mobility-dataset" + "/" + model_cat +
                       "/" + model_name + "/" + "textured_objs")

    visual_files = []
    for f in files:
        visual_files.append("./assets/partnet-mobility-dataset" + "/" +
                            model_cat + "/" + model_name + "/" +
                            "textured_objs/" + f)

    x_length, y_length, z_length, x_center, y_center, z_center = load_mesh(
        visual_files, quat=np.array(cat_orientation[model_cat]))
    material = scene.create_physical_material(100, 100, 0)

    if height is not None:
        scales = np.array([
            0.025 / x_length,
            0.025 / y_length,
            height / z_length,
        ])

    # if not visual_only:
    #     builder.add_multiple_collisions_from_file(
    #         filename="./assets/partnet-mobility-dataset" + "/" + model_cat +
    #         "/" + model_name + "/" + "convex_mesh.stl",
    #         scale=scales,
    #         density=1000,
    #         # material=material,
    #         pose=sapien.Pose(q=cat_orientation[model_cat]))

    for filename in visual_files:

        if not filename.endswith("obj"):
            continue

        builder.add_visual_from_file(
            filename=filename,
            scale=scales,
            pose=sapien.Pose(q=cat_orientation[model_cat]))

    if not visual_only:

        files = os.listdir("./assets/partnet-mobility-dataset" + "/" +
                           model_cat + "/" + model_name + "/" + "new_objs")

        visual_files = []
        for f in files:
            if not f.endswith("obj"):
                continue

            builder.add_multiple_collisions_from_file(
                filename="./assets/partnet-mobility-dataset" + "/" +
                model_cat + "/" + model_name + "/" + "new_objs/" + f,
                scale=scales,
                density=1000,
                pose=sapien.Pose(q=cat_orientation[model_cat]))

    if not visual_only:
        actor = builder.build()
    else:
        actor = builder.build_static(name=model_name)

    return actor, np.array(
        scales), abs(z_center - abs(z_length / 2)) * scales[2], np.array(
            [x_length, y_length, z_length]) * scales


USB_dict = {}

for model_name in (USB_ANYTRAIN + USB_ANYTRAIN_NOVEL):
    if model_name in USB_dict.keys():
        continue
    USB_dict[model_name] = {}
    USB_dict[model_name] = parse_urdf(
        "../sapien_task/assets/partnet-mobility-dataset/USB/" + model_name +
        "/mobility.urdf",
        USB_dict[model_name],
        model_name,
        model_cat="USB")


#USB_dict['100085']["joint"]["joint_0"]["limit"][1]-USB_dict['100085']["joint"]["joint_0"]["limit"][0]
#USB_dict['100085']["link"]["link_0"]["collision"]
def load_partnet_arti2actor(scene: sapien.Scene,
                            model_cat: str,
                            model_name: str,
                            physical_material: sapien.PhysicalMaterial = None,
                            density=1000,
                            visual_only=False,
                            height=None,
                            insertion=False):

    r = R.from_euler('xyz', [90, 0, 90], degrees=True)
    orientation = r.as_quat()[[3, 0, 1, 2]]

    cat_orientation = {"pen": orientation, "USB": [1, 0, 0, 0]}
    builder = scene.create_actor_builder()

    files = os.listdir("./assets/partnet-mobility-dataset" + "/" + model_cat +
                       "/" + model_name + "/" + "textured_objs")

    collision_files = []
    for f in files:
        collision_files.append("./assets/partnet-mobility-dataset" + "/" +
                               model_cat + "/" + model_name + "/" +
                               "textured_objs/" + f)

    x_length, y_length, z_length, x_center, y_center, z_center = load_mesh(
        collision_files, quat=np.array(cat_orientation[model_cat]))
    material = scene.create_physical_material(100, 100, 0)

    if height is not None:
        scales = np.array(
            [0.016 / x_length, 0.026 / y_length, (height) / z_length])

    if insertion:
        scales = np.array(
            [0.008 / x_length, 0.026 / y_length, (height) / z_length])
    # scales = np.ones(3)

    for filename in USB_dict[model_name]["link"]["link_0"][
            "collision"]:  # load link0 files
        if not filename.endswith("obj"):
            continue

        if not visual_only:
            builder.add_multiple_collisions_from_file(
                filename=filename,
                scale=scales,
                density=1000,
                # material=material,
                pose=sapien.Pose(p=[
                    0, 0,
                    (USB_dict[model_name]["joint"]["joint_0"]["limit"][0]) *
                    scales[2]
                ],
                                 q=cat_orientation[model_cat]))

    for filename in USB_dict[model_name]["link"]["link_0"][
            "visual"]:  # load link0 files
        builder.add_visual_from_file(
            filename=filename,
            scale=scales,
            # material=material,
            pose=sapien.Pose(p=[
                0, 0, (USB_dict[model_name]["joint"]["joint_0"]["limit"][0]) *
                scales[2]
            ],
                             q=cat_orientation[model_cat]))

    for filename in USB_dict[model_name]["link"]["link_1"][
            "collision"]:  # load link1 files
        if not filename.endswith("obj"):
            continue

        if not visual_only:
            builder.add_multiple_collisions_from_file(
                filename=filename,
                scale=scales,
                density=1000,
                # material=material,
                pose=sapien.Pose(p=[0, 0, 0], q=cat_orientation[model_cat]))

    for filename in USB_dict[model_name]["link"]["link_1"][
            "visual"]:  # load link1 files
        builder.add_visual_from_file(
            filename=filename,
            scale=scales,
            # material=material,
            pose=sapien.Pose(p=[0, 0, 0], q=cat_orientation[model_cat]))

    if not visual_only:
        actor = builder.build()
    else:
        actor = builder.build_static(name=model_name)

    return actor, np.array(
        scales), abs(z_center - abs(z_length / 2)) * scales[2], np.array(
            [x_length, y_length, z_length]) * scales


def load_partnet_articulation(
        scene: sapien.Scene,
        model_cat: str,
        model_name: str,
        physical_material: sapien.PhysicalMaterial = None,
        density=1000,
        visual_only=False):

    cat_orientation = {
        "pen": [0.707, 0.707, 0, 0],
        "knife": [1, 0, 0, 0],
        "USB": [1, 0, 0, 0]
    }
    cat_scale = {"pen": 0.7, "knife": 0.65, "USB": 0.75}

    if not visual_only:
        builder = scene.create_articulation_builder()
        partnet_link = builder.create_link_builder()
        handle_link = builder.create_link_builder()

    else:
        builder = scene.create_actor_builder()

    files = os.listdir("./assets/partnet-mobility-dataset" + "/" + model_cat +
                       "/" + model_name + "/" + "textured_objs")

    collision_files = []
    for f in files:
        collision_files.append("./assets/partnet-mobility-dataset" + "/" +
                               model_cat + "/" + model_name + "/" +
                               "textured_objs/" + f)

    scales = np.array([0.10, 0.10, 0.10]) * cat_scale[model_cat]

    x_length, y_length, z_length, x_center, y_center, z_center = load_mesh(
        collision_files, quat=np.array(cat_orientation[model_cat]))
    material = scene.create_physical_material(100, 100, 0)

    # scales = np.array([
    #     0.02547181 / x_length, 0.20568502 * 2 / 3 / y_length,
    #     0.02547181 / x_length
    # ])
    x_offset = 0.0000573 - x_length / 4 * scales[0]
    z_offset = 0.0000573 - z_length / 2 * scales[1]

    for filename in collision_files:
        if not filename.endswith("obj"):
            continue

        if not visual_only:

            partnet_link.add_multiple_collisions_from_file(
                filename=filename,
                scale=scales,
                density=1000,
                # material=material,
                pose=sapien.Pose(q=cat_orientation[model_cat]))

            handle_link.add_box_collision(
                half_size=[
                    x_length / 4 * scales[0] + x_offset,
                    z_length / 2 * scales[1] + z_offset, 0.01
                ],
                pose=sapien.Pose(
                    [0, 0,
                     abs(y_center - abs(y_length / 2)) * scales[1] / 2]),
                density=1000)

        if not visual_only:
            partnet_link.add_visual_from_file(
                filename=filename,
                scale=scales,
                pose=sapien.Pose(q=cat_orientation[model_cat]))

            partnet_link.set_name("object")

            handle_link.add_box_visual(
                half_size=[
                    x_length / 4 * scales[0] + x_offset,
                    z_length / 2 * scales[1] + z_offset, 0.01
                ],
                pose=sapien.Pose(
                    [0, 0,
                     abs(y_center - abs(y_length / 2)) * scales[1] / 2]))

            handle_link.set_name("handle")

            partnet_link.set_parent(handle_link.get_index())
        else:
            builder.add_visual_from_file(
                filename=filename,
                scale=scales,
                pose=sapien.Pose(q=cat_orientation[model_cat]))

            builder.add_box_visual(
                half_size=[
                    x_length / 4 * scales[0] + x_offset,
                    z_length / 2 * scales[1] + z_offset, 0.01
                ],
                pose=sapien.Pose(
                    [0, 0,
                     abs(y_center - abs(y_length / 2)) * scales[1] / 2]))

    if not visual_only:
        actor = builder.build(fix_root_link=False)
    else:
        actor = builder.build_static(name=model_name)

    if model_cat in ["knife"]:
        return actor, abs(z_center - abs(z_length / 2)) * scales[2], np.array(
            scales), np.array([x_length, y_length, z_length]) * scales

    else:

        return actor, scales, (
            abs(z_center) + z_length / 2) * scales[2], np.array(
                [x_length, y_length, z_length]) * np.array(scales) / 2


def load_car_object(scene: sapien.Scene,
                    model_name: str,
                    physical_material: sapien.PhysicalMaterial = None,
                    density=1000,
                    visual_only=False):

    # A heuristic way to infer split

    filename = "./assets/partnet-mobility-dataset/car/" + model_name + "/" + "mobility.urdf"
    loader = scene.create_urdf_loader()
    loader.scale = 0.2

    builder = loader.load_file_as_articulation_builder(filename)
    loader.load_multiple_collisions_from_file = True
    if not visual_only:
        # if model_name not in ["101052"]:
        object = builder.build()
        # else:
        #     object = builder.build()
    else:

        object = builder.build(fix_root_link=True)

    return object, loader.scale


# def load_pour_object(scene: sapien.Scene,
#                      model_name: str,
#                      physical_material: sapien.PhysicalMaterial = None,
#                      density=1000,
#                      visual_only=False):

#     # A heuristic way to infer split

#     filename = "./assets/partnet-mobility-dataset/pour/" + model_name + "/" + "mobility.urdf"
#     loader = scene.create_urdf_loader()
#     loader.scale = 0.04

#     builder = loader.load_file_as_articulation_builder(filename)
#     loader.load_multiple_collisions_from_file = True
#     # if not visual_only:
#     #     # if model_name not in ["101052"]:
#     #     object = builder.build()
#     #     # else:
#     #     #     object = builder.build()
#     # else:

#     object = builder.build()

#     # object.set_pose(sapien.Pose(q=[0.707, 0.707, 0, 0]))

#     return object, loader.scale

pour_dict = {}
POUR_ANYTRAIN = ["4084"]

for model_name in POUR_ANYTRAIN:
    if model_name in pour_dict.keys():
        continue
    pour_dict[model_name] = {}
    pour_dict[model_name] = parse_urdf(
        "../sapien_task/assets/partnet-mobility-dataset/pour/" + model_name +
        "/mobility.urdf", pour_dict[model_name], model_name, "pour")


def calculate_length(points):

    x_min, x_max = np.min(points[:, 0]), np.max(points[:, 0])
    y_min, y_max = np.min(points[:, 1]), np.max(points[:, 1])
    z_min, z_max = np.min(points[:, 2]), np.max(points[:, 2])

    return (x_max - x_min), (y_max - y_min), (
        z_max -
        z_min), (x_min + x_max) / 2, (y_min + y_max) / 2, (z_min + z_max) / 2


import trimesh
from scipy.spatial.transform import Rotation as R


def load_mesh(file, quat=None):
    points = []

    path = file

    if isinstance(file, list):
        points = []
        for f in file:

            if not (f.endswith("obj") or f.endswith("stl")):
                continue
            mesh = trimesh.load(f)

            if not isinstance(mesh, trimesh.Trimesh):
                for msh_key in mesh.geometry.keys():

                    msh = mesh.geometry[msh_key]
                    points.append(msh.vertices)
            else:
                point = mesh.vertices
                points.append(point)

    else:
        mesh = trimesh.load(path)

        if not isinstance(mesh, trimesh.Trimesh):

            for msh_key in mesh.geometry.keys():

                msh = mesh.geometry[msh_key]
                points.append(msh.vertices)

        else:
            point = mesh.vertices
            points.append(point)

    points = np.concatenate(points, axis=0)
    if quat is not None:

        r = R.from_quat(quat[[1, 2, 3, 0]])
        points = r.apply(points)

    x_length, y_length, z_length, x_center, y_center, z_center = calculate_length(
        points)

    return x_length, y_length, z_length, x_center, y_center, z_center


def load_pour_object(scene: sapien.Scene,
                     model_name: str,
                     physical_material: sapien.PhysicalMaterial = None,
                     object_cat="bottle",
                     density=1000,
                     visul_only=False):

    # A heuristic way to infer split

    builder = scene.create_articulation_builder()
    link = builder.create_link_builder()
    scales = [0.04, 0.10, 0.04]

    x_length, y_length, z_length, x_center, y_center, z_center = load_mesh(
        "./assets/partnet-mobility-dataset/" + object_cat + "/collision/" +
        model_name + "/link_1.stl")

    for collison_file in pour_dict[model_name]["link"]["link_1"]["collision"]:

        link.add_multiple_collisions_from_file(collison_file,
                                               scale=scales,
                                               density=2000)
    for visual_file in pour_dict[model_name]["link"]["link_1"]["visual"]:
        link.add_visual_from_file(visual_file, scale=scales)

    for link in builder.get_link_builders():
        link.set_collision_groups(1, 1, 4, 4)
    actor = builder.build(fix_root_link=False)

    return actor, scales, abs(y_center - y_length / 2) * scales[1] + 0.06


import os

SPOON_ORIENTATION = {
    # "spoon16": [-90, 0, 0],
    "spoon15": [90, 90, 0],
    "spoon14": [90, 90, 0],
    "spoon13": [90, 90, 0],
    "spoon12": [90, 90, 0],
    "spoon11": [90, 90, 0],
    "spoon10": [90, 90, 0],
    "spoon9": [90, 180, 0],
    # "spoon0": [0, -90, 0]
}

SPOON_ANYTRAIN = list(SPOON_ORIENTATION.keys())
SPOON_NOVEL_ANYTRAIN = list(SPOON_ORIENTATION.keys())


def load_spoon_object(scene: sapien.Scene,
                      model_name: str,
                      physical_material: sapien.PhysicalMaterial = None,
                      object_cat="spoon",
                      density=1000,
                      visual_only=False):

    # A heuristic way to infer split

    builder = scene.create_actor_builder()
    r = R.from_euler('xyz', SPOON_ORIENTATION[model_name], degrees=True)
    orientation = r.as_quat()[[3, 0, 1, 2]]
    pose = sapien.Pose(q=orientation)

    collision_files = os.listdir("./assets/spoon/collision_center/" +
                                 model_name)

    full_path_list = [
        os.path.join("./assets/spoon/collision_center/" + model_name, path2)
        for path2 in collision_files
    ]

    x_length, y_length, z_length, x_center, y_center, z_center = load_mesh(
        full_path_list, quat=pose.q)

    _, handle_length, _, _, _, z_lst_center = load_mesh(
        "./assets/spoon/collision_center/" + model_name + "/%d.stl" %
        (len(collision_files) - 1),
        quat=pose.q)

    length = 0.04573

    scales = np.array(
        [length / x_length, length / x_length, length / x_length])

    scales = np.array([0.018816935797248798 / handle_length] * 3)

    if not visual_only:

        builder.add_visual_from_file("./assets/spoon/visual_center/" +
                                     model_name + ".stl",
                                     scale=scales,
                                     pose=pose)

        for file in collision_files:

            builder.add_multiple_collisions_from_file(
                "./assets/spoon/collision_center/" + model_name + "/" + file,
                scale=scales,
                density=1000,
                pose=pose)
            # builder.add_visual_from_file(
            #     "./assets/spoon/visual_center/" + model_name+".stl",
            #     scale=scales,
            #     pose=pose)
    else:
        # for file in collision_files:
        # builder.add_visual_from_file(
        #     "./assets/spoon/collision_center/" + model_name + "/" + file,
        #     scale=scales,
        #     pose=sapien.Pose(
        #         p=[0, 0, ((abs(z_center) + z_length / 2) * scales[2]) * 0],
        #         q=pose.q))
        builder.add_visual_from_file("./assets/spoon/visual_center/" +
                                     model_name + ".stl",
                                     scale=scales,
                                     pose=pose)

    # link.add_visual_from_file("./assets/spoon/visual/" + model_name + ".stl",
    #                           scale=scales)

    if not visual_only:
        actor = builder.build()
    else:
        actor = builder.build_static()

    return actor, scales, (abs(z_center) + z_length / 2) * scales[2], np.array(
        [x_length, y_length, z_length]) * np.array(scales) / 2


# KITCHEN_ANYTRAIN = ["spoon" + str(i) for i in range(16, 27)]
# KITCHEN_ANYTRAIN.remove("spoon24")
# print(KITCHEN_ANYTRAIN)
KITCHEN_ANYTRAIN = [
    'spoon16', 'spoon17', 'spoon18', 'spoon20', 'spoon22', 'spoon27',
    "spoon31", "spoon32", "spoon33", "spoon34", "spoon35", "spoon36",
    "spoon37", "spoon38", "spoon41", "spoon42", "spoon43"
]

KITCHEN_ORIENTATION = {
    "spoon22": [90, 0, 0],
    "spoon24": [90, 0, 0],
    "spoon25": [90, 0, 0],
    "spoon27": [90, 0, 0],
    "spoon29": [90, 0, 0],
}
KITCHEN_ANYTRAIN = [
    'spoon22', 'spoon27', "spoon31", "spoon32", "spoon33", "spoon34",
    "spoon35", "spoon41", "spoon43"
]


def load_kitchen_object(scene: sapien.Scene,
                        model_name: str,
                        physical_material: sapien.PhysicalMaterial = None,
                        object_cat="spoon",
                        density=1000,
                        visual_only=False):

    builder = scene.create_actor_builder()
    if model_name not in KITCHEN_ORIENTATION.keys():
        r = R.from_euler('xyz', [-90, 0, 0], degrees=True)
    else:
        r = R.from_euler('xyz', KITCHEN_ORIENTATION[model_name], degrees=True)
    orientation = r.as_quat()[[3, 0, 1, 2]]
    pose = sapien.Pose(q=orientation)

    collision_files = os.listdir("./assets/spoon/collision_center/" +
                                 model_name)

    full_path_list = [
        os.path.join("./assets/spoon/collision_center/" + model_name, path2)
        for path2 in collision_files
    ]

    x_length, y_length, z_length, x_center, y_center, z_center = load_mesh(
        full_path_list, quat=pose.q)

    _, handle_length, _, _, _, z_lst_center = load_mesh(
        "./assets/spoon/collision_center/" + model_name + "/%d.stl" %
        (len(collision_files) - 1),
        quat=pose.q)

    length = 0.08573

    scales = np.array(
        [length / x_length, length / x_length, length / x_length])

    scales = np.array([0.008816935797248798 / handle_length] * 3)

    height = 0.02
    width = 0.00873

    if not visual_only:

        for file in collision_files:

            builder.add_multiple_collisions_from_file(
                "./assets/spoon/collision_center/" + model_name + "/" + file,
                scale=scales,
                density=1000,
                pose=pose)
            builder.add_visual_from_file("./assets/spoon/collision_center/" +
                                         model_name + "/" + file,
                                         scale=scales,
                                         pose=pose)
        # builder.add_box_collision(
        #     half_size=[width, width, height],
        #     pose=sapien.Pose(
        #         [0, 0, (abs(z_center) + z_length / 2) * scales[2] * 0]),
        #     density=1000)
        # builder.add_box_visual(
        #     half_size=[width, width, height],
        #     pose=sapien.Pose(
        #         [0, 0, (abs(z_center) + z_length / 2) * scales[2] * 0]),
        # )
    else:
        for file in collision_files:
            builder.add_visual_from_file("./assets/spoon/collision_center/" +
                                         model_name + "/" + file,
                                         scale=scales,
                                         pose=pose)
        # builder.add_box_visual(
        #     half_size=[width, width, height],
        #     pose=sapien.Pose(
        #         [0, 0, (abs(z_center) + z_length / 2) * scales[2] * 0]),
        # )

    # link.add_visual_from_file("./assets/spoon/visual/" + model_name + ".stl",
    #                           scale=scales)

    if not visual_only:
        actor = builder.build()
    else:
        actor = builder.build_static()

    return actor, scales, (abs(z_center) + z_length / 2) * scales[2], np.array(
        [x_length, y_length, z_length]) * np.array(scales) / 2
