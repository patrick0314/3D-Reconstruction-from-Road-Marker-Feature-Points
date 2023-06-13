import pickle
import time

import numpy as np
import open3d as o3d
from matplotlib.pylab import *

# print("testing o3d...")

# pcd_file = "../../ITRI_dataset/seq1/dataset/1681710717_532211005/sub_map.csv"
# pcd_array = np.loadtxt(pcd_file, dtype=float, delimiter=",")

# pcd = o3d.geometry.PointCloud()
# pcd.points = o3d.utility.Vector3dVector(pcd_array)
# o3d.visualization.draw_geometries([pcd])

def visualize_pcd(pcd, color=None):
    """
        input: pcd, str or 2d numpy array with [x,y,z]
    """
    if type(pcd) == str:
        pcd = np.loadtxt(pcd, dtype=float, delimiter=",")
    vpcd = o3d.geometry.PointCloud()
    vpcd.points = o3d.utility.Vector3dVector(pcd)
    if color is not None:
        print(type(color[0,0]))
        color = color.astype(np.float64)/255.0
        vpcd.colors = o3d.utility.Vector3dVector(color)
    # o3d.visualization.draw_geometries([vpcd])
    viewer = o3d.visualization.Visualizer()
    viewer.create_window()
    for geometry in [vpcd]:
        viewer.add_geometry(geometry)
    opt = viewer.get_render_option()
    opt.show_coordinate_frame = True
    opt.background_color = np.asarray([0, 0, 0])
    viewer.run()

def display_points3d(tripoints3d, pcd, visualizer):
    # open3d
    if tripoints3d is not None:
        pcd.clear()
        visualizer.remove_geometry(pcd)
        pcd.points = o3d.utility.Vector3dVector(tripoints3d)
        visualizer.add_geometry(pcd)
        visualizer.poll_events()
        visualizer.update_renderer()
        time.sleep(.2)

if __name__ == "__main__":
    # pcd = "../../ITRI_dataset/seq1/dataset/1681710722_331000126/sub_map.csv"
    # visualize_pcd(pcd)
    with open("pcd_map.pckl","rb") as f:
        map, color = pickle.load(f)
    print(color.shape[0])
    len0 = color.shape[0]
    print(map.shape[0])
    # map[:,2] = -map[:,2]
    over_road_mask = np.where(map[:,2] < -1.5)
    visualize_pcd(map, color)