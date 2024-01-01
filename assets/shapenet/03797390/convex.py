import numpy as np
import trimesh
import os
from pathlib import Path
import pdb
import pandas as pd
import shutil
import pickle


def read_materials(filename):
    mats = []
    mtl = None  # current material
    for line in open(filename, "r"):
        if line.startswith('#'):
            continue
        values = line.split()
        if not values:
            continue
        if values[0] == 'newmtl':
            if mtl is not None:
                mats.append(mtl)
            mtl = {}
            mtl['name'] = values[1]
        elif values[0] == 'Kd':
            mtl['Kd'] = " ".join(values[1:])

    if mtl is not None:
        mats.append(mtl)

    return pd.DataFrame(mats)


obj_list = os.listdir("/home/lme/Downloads/03797390")

dict = {}

for obj in obj_list:
    if not os.path.isdir("/home/lme/Downloads/03797390/" + obj):
        continue

    result_path = Path("/home/lme/Downloads/03797390/" + obj + "/collision")
    result_path.mkdir(exist_ok=True, parents=True)

    msh = trimesh.load("/home/lme/Downloads/03797390/" + obj + "/model.obj")

    transformation_matrix = np.identity(4)

    # Collect all vertices from the scene
    all_vertices = []

    if isinstance(msh, trimesh.Scene):
        for mesh in msh.geometry.keys():
            # Get the vertices of the current mesh

            vertices = msh.geometry[mesh].vertices
            # Append the vertices to the list
            all_vertices.extend(vertices)

        all_vertices = np.array(all_vertices)
    else:
        all_vertices = np.array(msh.vertices)

    # transformation_matrix[0, 3] = -np.max(all_vertices[:, 0])

    # for mesh in msh.geometry:
    #     # Apply the transformation to the mesh vertices
    #     mesh.apply_transform(transformation_matrix)

    # msh.apply_transform(transformation_matrix)
    # msh.export("/home/lme/Downloads/03797390/" + obj + "/visual.obj")

    obj_frame = read_materials("/home/lme/Downloads/03797390/" + obj +
                               "/model.obj")

    convex_msh = trimesh.decomposition.convex_decomposition(msh, )

    if isinstance(convex_msh, trimesh.base.Trimesh):
        shutil.rmtree("/home/lme/Downloads/03797390/" + obj)
        continue
    print(obj)
    dict[obj] = -np.max(all_vertices[:, 0])

    for index, m in enumerate(convex_msh):
        m.export("/home/lme/Downloads/03797390/" + obj +
                 "/collision/%d.obj" % index)

with open("mug.pkl", "wb") as file:
    pickle.dump(dict, file)
