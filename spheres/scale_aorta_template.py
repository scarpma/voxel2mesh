import numpy as np
import pyvista as pv
import os.path as osp

templateName = 'aorta_template_V2'
templatePath = osp.join('spheres', templateName + '.vtk')
m = pv.read(templatePath)
m = m.decimate(0.93)

maxs = m.points.max(axis=0)
mins  = m.points.min(axis=0)

m.points = (m.points - mins) / (maxs - mins)
m.points = m.points*2 - 1
m = m.triangulate()
m.save(osp.join('spheres', templateName + '_proc' + '.vtk'))
