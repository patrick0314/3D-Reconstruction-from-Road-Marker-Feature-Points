import cv2
from visualization import OdomPlot
from VisualOdometry import VisualOdometry
import config
import matplotlib.pyplot as plt
from o3dUtils import visualize_pcd, display_points3d
import pickle
import open3d as o3d
from dataset2numpy import Dataset
import numpy as np
from ICP import ICP_all_timestamp
from eval import benchmark
from Utility import ext_cam

OdomPlt = OdomPlot()
seq = "seq1"
dataset = Dataset(seq, loc="../../ITRI_dataset/")
all_timestamp = dataset.get_time_stamp()
direction = dataset.get_camera_dir()
raw_image_front = dataset.get_raw_image([all_timestamp[i] for i in np.where(direction == 0)[0]])
print(raw_image_front.shape)    # shape: (total images num, height, width, channel)
frame_num, _, _, _ = raw_image_front.shape

Front_VO = VisualOdometry(config.Front_Cam,
                    config.feature_finder,
                    config.descriptors_computer,
                    config.matcher,
                    config.findEssentialMat,
                    isDebug=config.VO_Debug)

# 3D shader
pcd = o3d.geometry.PointCloud()
viewer = o3d.visualization.Visualizer()
viewer.create_window(width=960, height=540)

opt = viewer.get_render_option()
opt.show_coordinate_frame = True
opt.background_color = np.asarray([0, 0, 0])
# 

Debug_ShowRoute = True
isFirstFrame = True
plt_pos = []
for i in range(frame_num):
    frame = raw_image_front[i]

    #Alert, 0 need to set to actual timestamp, timestamp not done yet!
    Front_VO.computeAndUpdateOdom(frame, 0)
    
    # skip first frame
    if isFirstFrame:
        isFirstFrame = False
        continue

    # Debugging
    if config.VO_Debug:
        frame = Front_VO.debugFrame[0]

    # Optional: Just to make it smaller so that it fit on my monitor.
    frame = cv2.resize(frame, ( 640,  480))
    
    # Draw the current frame Speed on screen
    position,_ = Front_VO.Odom.getPoseByEuclidean()
    x, y, z = position
    cv2.putText(frame, "x: "+str(x), (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(frame, "y: "+str(y), (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(frame, "z: "+str(z), (30, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # record the route
    plt_pos.append([x,z])
    display_points3d(Front_VO.map_tmp ,pcd, viewer)
    # mean_dist = np.mean(np.linalg.norm(Front_VO.map, 2, axis=1))
    # print(mean_dist)
    # show features
    cv2.imshow('Odometry', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()

plt_pos = np.array(plt_pos)
plt.plot(plt_pos[:,0], plt_pos[:,1])
plt.show()

color = Front_VO.o3dcolor()

pcd_map = Front_VO.map
pcd_map = np.vstack((pcd_map.T, np.ones(pcd_map.shape[0]).reshape(1, -1)))
pcd_map = (ext_cam.f_c.get_Transform_to_baselink() @ pcd_map) [:3].T
# pcd_map[:,[1,2]] = -pcd_map[:,[1,2]]
if Debug_ShowRoute:
    pcd_route = Front_VO.route
    pcd_route = np.vstack((pcd_route.T, np.ones(pcd_route.shape[0]).reshape(1, -1)))
    pcd_route = (ext_cam.f_c.get_Transform_to_baselink() @ pcd_route) [:3].T
    color = np.concatenate((color, np.repeat(np.array([0,0,255]).reshape(1,3), pcd_route.shape[0], axis=0)), axis=0)
    pcd_map = np.concatenate((pcd_map, pcd_route), axis=0)
with open("pcd_map.pckl", "wb") as f:
    pickle.dump([pcd_map, color], f)

result = ICP_all_timestamp(pcd_map, seq=seq, threshold=2)
np.savetxt("pred_pose.txt", result, delimiter=" ")
seq=["seq1"]
dataset_path = "../../ITRI_dataset"
benchmark(dataset_path, seq, src_path="./pred_pose.txt")
# print(pcd_map)
# cv2.imshow("image", raw_image_front[79])
visualize_pcd(pcd_map, color)
cv2.waitKey(0)
