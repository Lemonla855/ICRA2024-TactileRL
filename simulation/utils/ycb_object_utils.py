from pathlib import Path

import numpy as np
import sapien.core as sapien

#===========================any========================
YCB_OBJECT_NAMES_EXIT = [
    "002_master_chef_can",
    # "004_sugar_box",
    "005_tomato_soup_can",
    # "006_mustard_bottle",
    "010_potted_meat_can",
    # "021_bleach_cleanser",
    # "025_mug",  #"051_large_clamp"
]
YCB_OBJECT_NAMES_EXIT_LIST = [
    "master_chef_can", "sugar_box", "mustard_bottle", "bleach_cleanser"
]
#===========================any========================


def get_ycb_root_dir():
    current_dir = Path(__file__).parent
    ycb_dir = current_dir.parent.parent / "assets" / "ycb"
    return ycb_dir.resolve()


YCB_CLASSES = {
    1: '002_master_chef_can',
    2: '003_cracker_box',
    3: '004_sugar_box',
    4: '005_tomato_soup_can',
    5: '006_mustard_bottle',
    6: '007_tuna_fish_can',
    7: '008_pudding_box',
    8: '009_gelatin_box',
    9: '010_potted_meat_can',
    10: '011_banana',
    11: '019_pitcher_base',
    12: '021_bleach_cleanser',
    13: '024_bowl',
    14: '025_mug',
    15: '035_power_drill',
    16: '036_wood_block',
    17: '037_scissors',
    18: '040_large_marker',
    19: '051_large_clamp',
    20: '052_extra_large_clamp',
    21: '061_foam_brick',
}

YCB_SIZE = {
    "master_chef_can": (0.1025, 0.1023, 0.1401),
    "cracker_box": (0.2134, 0.1640, 0.0717),
    "sugar_box": (0.0495, 0.0940, 0.1760),
    "tomato_soup_can": (0.0677, 0.0679, 0.1018),
    "mustard_bottle": (0.0576, 0.0959, 0.1913),
    "potted_meat_can": (0.0576, 0.1015, 0.0835),
    "banana": (0.1088, 0.1784, 0.0366),
    "bleach_cleanser": (0.1024, 0.0677, 0.2506),
    "bowl": (0.1614, 0.1611, 0.0550),
    "mug": (0.1169, 0.0930, 0.0813),
    "large_clamp": (0.1659, 0.1216, 0.0364),
}

YCB_ORIENTATION = {
    "master_chef_can": (1, 0, 0, 0),
    "cracker_box": (1, 0, 0, 0),
    "sugar_box": (1, 0, 0, 0),
    "tomato_soup_can": (1, 0, 0, 0),
    # "mustard_bottle": (0.9659, 0, 0, 0.2588),
    "mustard_bottle": (0.500, 0, 0, 0.866),
    "potted_meat_can": (1, 0, 0, 0),
    # "potted_meat_can": (0.77, 0, 0, 0.77),
    "banana": (1, 0, 0, 0),
    "bleach_cleanser": (1, 0, 0, 0),
    "bowl": (1, 0, 0, 0),
    "mug": (1, 0, 0, 0),
    "large_clamp": (0, 0, 0, 1),
}

INVERSE_YCB_CLASSES = {
    "_".join(value.split("_")[1:]): key
    for key, value in YCB_CLASSES.items()
}
YCB_OBJECT_NAMES = list(INVERSE_YCB_CLASSES.keys())
YCB_ROOT = get_ycb_root_dir()
YCB_BBPoint = {
    '021_bleach_cleanser': np.array([-0.057088, -0.032301, -0.112916]),
    '051_large_clamp': np.array([-0.07176, -0.085225, -0.020491]),
    '004_sugar_box': np.array([-0.024896, -0.049405, -0.093333]),
    '002_master_chef_can': np.array([-0.053107, -0.051235, -0.074167]),
    '025_mug': np.array([-0.0455, -0.0455, -0.038487]),
    '010_potted_meat_can': np.array([-0.054221, -0.031714, -0.050578]),
    '061_foam_brick': np.array([-0.026684, -0.039597, -0.017837]),
    '005_tomato_soup_can': np.array([-0.035018, -0.033134, -0.060134]),
    '006_mustard_bottle': np.array([-0.050772, -0.036723, -0.084818]),
    '011_banana': np.array([-0.036589, -0.102265, -0.017718])
}


def load_ycb_object(scene: sapien.Scene,
                    object_name,
                    scale=1,
                    visual_only=False,
                    material=None,
                    static=False):

    ycb_id = INVERSE_YCB_CLASSES[object_name]
    ycb_name = YCB_CLASSES[ycb_id]
    visual_file = YCB_ROOT / "visual" / ycb_name / "textured_simple.obj"
    collision_file = YCB_ROOT / "collision" / ycb_name / "collision.obj"
    builder = scene.create_actor_builder()
    scales = np.array([scale] * 3)
    density = 1000
    if material is None:
        material = scene.engine.create_physical_material(100000000, 0.5, 1)

    if not visual_only:
        builder.add_multiple_collisions_from_file(str(collision_file),
                                                  scale=scales,
                                                  density=density,
                                                  material=material)
    if visual_only:
        visual_file = YCB_ROOT / "visual" / ycb_name / "textured_simple.stl"

    builder.add_visual_from_file(str(visual_file), scale=scales)
    if not visual_only and not static:
        actor = builder.build(name=YCB_CLASSES[ycb_id])
    else:
        actor = builder.build_static(name=f"{YCB_CLASSES[ycb_id]}_visual")
    return actor


def load_ycb_articulated_object(scene: sapien.Scene,
                                object_name,
                                scale=1,
                                visual_only=False,
                                material=None,
                                static=False):
    builder: sapien.ArticulationBuilder = scene.create_articulation_builder()
    obj_link = builder.create_link_builder()

    ycb_id = INVERSE_YCB_CLASSES[object_name]
    ycb_name = YCB_CLASSES[ycb_id]
    obj_link.add_multiple_collisions_from_file(
        str("assets/ycb/collision/" + ycb_name + "/collision.obj"),
        scale=np.array([1., 1., 1.]))
    obj_link.add_visual_from_file(str("assets/ycb/visual/" + ycb_name +
                                      "/textured_simple.obj"),
                                  scale=np.array([1., 1., 1.]))

    slider_link = builder.create_link_builder()

    bbox_points = YCB_BBPoint[ycb_name]

    obj_link.set_joint_properties(
        joint_type="revolute",
        limits=[[-np.pi, np.pi]],
        pose_in_child=sapien.Pose(
            [bbox_points[0], bbox_points[1], bbox_points[2]], [1, 0, 0, 0]),
        pose_in_parent=sapien.Pose(
            [bbox_points[0], bbox_points[1], bbox_points[2]], [1, 0, 0, 0]))

    obj_link.set_parent(slider_link.get_index())
    collision = builder.build(fix_root_link=True)

    collision.get_active_joints()[-1].set_drive_property(200,
                                                         60,
                                                         10,
                                                         mode="force")
    collision.set_qpos([np.pi / 2])
    collision.set_drive_target([np.pi / 2])

    return collision
