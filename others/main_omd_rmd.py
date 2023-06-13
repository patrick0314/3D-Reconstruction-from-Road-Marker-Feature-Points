import cv2
from visualization import OdomPlot
from VisualUtils import ShortTimeReconstruct, kp2np
import config
import matplotlib.pyplot as plt
from o3dUtils import visualize_pcd, display_points3d
import pickle
import open3d as o3d
from dataset2numpy import Dataset
import numpy as np
from ICP import ICP_all_timestamp, ICP_with_timestamp
from eval import benchmark
from Utility import ext_cam
from RoadMarker import RoadMarkerDetection

seq = "seq1"
kptsDebug = True # look keypnts matching

# init dataset
dataset = Dataset(seq, loc="../../ITRI_dataset/")
all_timestamp = dataset.get_time_stamp()
local_timestamp = dataset.get_time_stamp(with_GT=True)
direction = dataset.get_camera_dir()
raw_image_front = dataset.get_raw_image([all_timestamp[i] for i in np.where(direction == 0)[0]])
frame_num, _, _, _ = raw_image_front.shape
road_marker_list = dataset.get_detect_road_marker()
front_road_marker_list = [road_marker_list[i] for i in np.where(direction == 0)[0]]
front_time_stamp = [all_timestamp[i] for i in np.where(direction == 0)[0]]

# init shortTimeReconstruct
Front_STR = ShortTimeReconstruct(
            config.Front_Cam,
            config.feature_finder,
            config.descriptors_computer,
            config.matcher,
            config.findEssentialMat,
)

# init Road Marker Detection
RMD = RoadMarkerDetection()

# 3D shader
pcd = o3d.geometry.PointCloud()
viewer = o3d.visualization.Visualizer()
viewer.create_window(width=960, height=540)

opt = viewer.get_render_option()
opt.show_coordinate_frame = True
opt.background_color = np.asarray([0, 0, 0])


# reconstruct for every 15 images, compare each frame with the center frame
overlapping = 7
center = 7
mask = Front_STR._camera.mask
idle_projection_matrix = Front_STR._camera.projection_matrix
camera_matrix = Front_STR._camera.camera_Matrix
while True:
    
    frames_to_reconstruct = raw_image_front[center-7: min(center+7+1, frame_num)]
    center_frame = raw_image_front[center]
    Front_STR.setCenterImage(center_frame, mask)

    # record road marker of center
    gray_center = cv2.cvtColor(center_frame, cv2.COLOR_BGR2GRAY)
    kp_road_center,  _ , _ = RMD.detection(center_frame.copy(), mask, front_road_marker_list[center], config)
    kp_road_center, descriptor_center = config.descriptors_computer(gray_center, kp_road_center)
    Front_STR.setCenterKptsnDes(kp_road_center, descriptor_center)
    proj_center = idle_projection_matrix
    pcd = np.zeros((0,3))
    for i in range(1+2*overlapping):
        if i == overlapping:
            continue
        
        curr_idx = center- overlapping+ i

        curr_frame = frames_to_reconstruct[i]
        gray_curr = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
        
        # record road marker of current frame
        if len(front_road_marker_list[curr_idx]) == 0:
            print("No marker list")
            continue
        kp_road_curr, _, _ = RMD.detection(frames_to_reconstruct[i].copy(), mask, front_road_marker_list[curr_idx], config)
        kp_road_curr, descriptor_curr = config.descriptors_computer(curr_frame.copy(), kp_road_curr)
        
        # use road marker to get translation
        # proj_curr, kp_road_center_more_matching, kp_road_curr_more_matching, matches = Front_STR.motionsFromKptsnDes(kp_road_curr, descriptor_curr, camera_matrix=camera_matrix, p_center=proj_center)

        proj_curr, _, _ = Front_STR.motionsFromFrame(curr_frame, camera_matrix=camera_matrix, p_center=proj_center)
        # match two kpts
        # matches_kp_road_more = config.matcher(descriptor_center, descriptor_curr)
        # matches_kp_road_less = config.matcher(descriptor_center_less, descriptor_curr_less)
        matches = config.matcher(descriptor_center, descriptor_curr)

        if kptsDebug:
            print("len kp curr:", len(kp_road_curr))
            print("len kp cent:", len(kp_road_center))
            debug_frame = cv2.drawMatches(center_frame, kp_road_center, curr_frame, kp_road_curr, 
                            matches, None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
            debug_frame = cv2.resize(debug_frame, ( 960,  540))
            cv2.imshow("debug",debug_frame)
            if cv2.waitKey(500) & 0xFF == ord('q'):
                cv2.waitKey(0)

        # proj_curr = proj_center @ (Transform_Matrix)

        # reconstruct road markers
        # kp to numpy        
        kp_road_center_np, kp_road_curr_np, _ = kp2np(kp_road_center, kp_road_curr, matches)

        if kp_road_center_np.size == 0:
            print("No matching point!!", curr_idx)
            continue

        pcd_tmp = Front_STR.reconstruct(
            kp_road_center_np,
            kp_road_curr_np,
            proj_center,
            proj_curr,
            )
        print("pcd iter:", pcd_tmp.shape)
        pcd = np.vstack((pcd, pcd_tmp))
    pcd = np.hstack((pcd, np.ones(pcd.shape[0]).reshape(-1,1))).T
    pcd = ext_cam.f_c.get_Transform_to_baselink() @ pcd
    pcd /= pcd[3]
    # visualize_pcd(pcd[:3].T)
    
    # check ICP
    test_timestamp = local_timestamp[11]
    print(front_time_stamp[7])
    print(test_timestamp)
    print(ICP_with_timestamp(pcd[:3].T, seq, test_timestamp, 1))
    
    
    break
    center += overlapping
    

