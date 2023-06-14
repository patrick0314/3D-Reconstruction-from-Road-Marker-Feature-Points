import numpy as np
import open3d as o3d
from config import dataset_path
import os

def ICP(source, target, threshold, init_pose, iteration=30):
    # implement iterative closet point and return transformation matrix
    reg_p2p = o3d.pipelines.registration.registration_icp(
        source, target, threshold, init_pose,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=iteration)
    )
    # print(reg_p2p)
    # assert len(reg_p2p.correspondence_set) != 0, 'The size of correspondence_set between your point cloud and sub_map should not be zero.'
    no_correspondence = len(reg_p2p.correspondence_set) == 0
    # print(reg_p2p.transformation)
    return reg_p2p.transformation, no_correspondence

def csv_reader(filename):
    # read csv file into numpy array
    data = np.loadtxt(filename, delimiter=',')
    return data

def numpy2pcd(arr):
    # turn numpy array into open3d point cloud format
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(arr)
    return pcd

def ICP_all_timestamp(src_npy, threshold, seq="seq1"):
    path_name = os.path.join(dataset_path, seq, "dataset")
    timestamp_path = os.path.join(dataset_path, seq, "localization_timestamp.txt")
    timestamp = np.loadtxt(timestamp_path, dtype=str)
    # init_pose = csv_reader(f'{path_name}initial_pose.csv')
    src_pcd = numpy2pcd(src_npy)
    result = []
    for i in timestamp:
        target_pcd = np.loadtxt(os.path.join(path_name, i, "sub_map.csv"), delimiter=",", dtype=float)
        target_pcd = numpy2pcd(target_pcd)
        init_pose = csv_reader(os.path.join(path_name, i, "initial_pose.csv"))
        transformation = ICP(src_pcd, target_pcd, threshold=threshold, init_pose=init_pose)
        result.append([transformation[0,3], transformation[1,3]])
    return np.array(result)

def ICP_with_timestamp(src_npy, seq, timestamp, threshold=0.02):
    src_pcd = numpy2pcd(src_npy)
    path_name = os.path.join(dataset_path, seq, "dataset", timestamp)
    target_pcd = np.loadtxt( os.path.join(path_name, "sub_map.csv"), delimiter=",", dtype=float)
    target_pcd = numpy2pcd(target_pcd)
    init_pose = csv_reader( os.path.join(path_name, "initial_pose.csv"))
    transformation, no_correspondence = ICP(src_pcd, target_pcd, threshold=threshold, init_pose=init_pose)
    return [transformation[0,3], transformation[1,3]], no_correspondence

    