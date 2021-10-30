import numpy as np
import glob
import os.path as osp
import os
import pyvista as pv
import trimesh
import pyacvd
import pandas as pd

from pymeshfix._meshfix import PyTMesh
from pymeshfix import MeshFix

DICOM_DIR = './data/'
oDir = osp.join(DICOM_DIR, '..', 'vox2MeshDataset')
num_points = 18000


def getRoiInfo(patName):
    path = osp.join(DICOM_DIR, 'ROI_arrays', f'{patName}.npy')
    array = np.load(path, allow_pickle=True)[()]
    roi_bounds = array['roi_bounds']
    vxSize_xyz = array['vxSize_xyz']
    origin_xyz = array['origin_xyz']
    return roi_bounds, np.array(vxSize_xyz), np.array(origin_xyz,
                                                      dtype=np.float)


def getName(string):
    return string.split('_')[0]


def pvToTrimesh(pvMesh):
    faces = pvMesh.faces.reshape((-1, 4))
    faces = faces[:, -3:]
    trimeshMesh = trimesh.Trimesh(
        vertices=pvMesh.points,
        faces=faces,
    )
    #n_faces = pvMesh.faces.shape[0] // 4
    #idxs = np.array([False, True, True, True]*n_faces)
    #faces = pvMesh.faces[idxs]
    #trimeshMesh = trimesh.Trimesh(
    #    vertices = pvMesh.points,
    #    faces = faces.reshape(n_faces,3),
    #)
    return trimeshMesh


def trimeshToPv(trimeshMesh):
    faces = trimeshMesh.faces
    tria = np.empty((faces.shape[0], 4), dtype=faces.dtype)
    tria[:, -3:] = faces
    tria[:, 0] = 3
    pvMesh = pv.PolyData(trimeshMesh.vertices, tria)
    #n_faces = trimeshMesh.faces.shape[0]
    #temp = np.array([3]*n_faces)
    #faces = np.c_[temp, trimeshMesh.faces]
    #pvMesh = pv.PolyData(trimeshMesh.vertices, faces.reshape(-1))
    return pvMesh


if not osp.exists(oDir):
    os.makedirs(oDir)

paths = glob.glob('*.stl')
paths = sorted(paths, key=getName)

for path in paths:
    print(path)
    patName = getName(path)
    m = pv.read(path)

    ### CLIPPING AND CONNECTED COMPONENT ANALYSIS
    perm = np.array([4, 5, 2, 3, 0, 1])
    roiBounds, vxSize_xyz, origin_xyz = getRoiInfo(patName)
    roiBounds = np.array(roiBounds)[perm]
    perm = np.array([0, 0, 1, 1, 2, 2])
    roi_vxSize = vxSize_xyz[perm]
    origin = origin_xyz[perm]

    boxBounds = roiBounds * roi_vxSize + origin
    b = pv.Box(boxBounds, level=2)
    b.save(osp.join(oDir, getName(path) + '_box.vtk'))
    print(boxBounds)
    m = m.clip_box(bounds=boxBounds, invert=False)
    m = m.extract_largest()
    m = m.extract_surface()
    m.clear_arrays()
    m = m.triangulate()

    mfix = PyTMesh(False)  # False removes extra verbose output
    mfix.load_array(m.points, m.faces.reshape((-1, 4))[:, -3:])
    mfix.fill_small_boundaries(nbe=1e6, refine=True)
    vert, faces = mfix.return_arrays()
    tria = np.empty((faces.shape[0], 4), dtype=faces.dtype)
    tria[:, -3:] = faces
    tria[:, 0] = 3
    m = pv.PolyData(vert, tria)

    ## ### SMOOTHING
    ## m = pvToTrimesh(m)
    ## lap = trimesh.smoothing.laplacian_calculation(m, equal_weight=False)
    ## m = trimesh.smoothing.filter_taubin(m, lamb=0.5, nu=0.5, iterations=50, laplacian_operator=lap)
    ## m = trimeshToPv(m)

    ### REMESHING
    clus = pyacvd.Clustering(m)
    #clus.subdivide(2)
    clus.cluster(num_points)
    m = clus.create_mesh()

    m.save(osp.join(oDir, getName(path) + '.vtk'))
