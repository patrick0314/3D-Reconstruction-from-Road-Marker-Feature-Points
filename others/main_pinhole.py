import cv2
import config
import matplotlib.pyplot as plt
from o3dUtils import visualize_pcd, display_points3d
from dataset2numpy import Dataset
import numpy as np
from Utility import ext_cam
from RoadMarker import RoadMarkerDetection
from Utility import Transform
from Utility import NormalizePoint, NormalizeImageIndex, NormalizeMatrix
import matplotlib.pyplot as plt
from pinhole import Pinhole, visualize_pcd_plt, visualize_pcd_plt_4cam
from ICP import ICP_with_timestamp

# take data from dataset
seq = "seq1"
dataset = Dataset(seq, loc="../../ITRI_dataset/")
all_timestamp = dataset.get_time_stamp()
local_timestamp = dataset.get_time_stamp(with_GT=True)
required_timestamp = dataset.get_4_cam_at_ts(local_timestamp[-1])
direction = dataset.get_camera_dir(required_timestamp)
print(required_timestamp)
raw_image = dataset.get_raw_image(required_timestamp)
road_marker_list = dataset.get_detect_road_marker(required_timestamp)

# order of direction at dataset: fl ->f ->b ->fr  1, 0, 3, 2 

# camera
f_c = Pinhole(config.Front_Cam, ext_cam.f_c, )
f_r = Pinhole(config.Right_Cam, ext_cam.f_r, )
f_l = Pinhole(config.Left_Cam,  ext_cam.f_l, )
f_b = Pinhole(config.Back_Cam,  ext_cam.b,   )

cameras = {0:f_c, 1:f_l, 2:f_r, 3:f_b}
masks = {0: f_c._camera.mask, 1: f_l._camera.mask, 2: f_r._camera.mask, 3: f_b._camera.mask}
colors = {0: "red", 1:"green",2:"blue", 3:"black"}
# road markers
RMD = RoadMarkerDetection(0.005)

pcd_four_cam = []
imgsKpts = []
pcd_o3d = np.zeros((0,3))
for i in range(4):
    camera = cameras[direction[i]]
    mask = masks[direction[i]]
    frame = raw_image[i]
    road_marker = road_marker_list[i]
    if len(road_marker) == 0:
        print("No roadmarker")
        continue
    
    kp_road, imgKpts, _ = RMD.detection(frame.copy(), mask, road_marker, config)
    # imgKpts = cv2.resize(imgKpts, (640, 640))
    imgsKpts.append(imgKpts)
    kp_road = np.float32([kp.pt for kp in kp_road])
    if kp_road.size == 0:
            print("\nNo kp!!", required_timestamp[i])
            continue
    pcd = camera.reconstruct_from_kpts_plane(kp_road)
    pcd_four_cam.append({"pcd":pcd, "color":colors[direction[i]]})
    pcd_o3d = np.vstack((pcd_o3d, pcd))
    # print(direction[i], pcd)
for idx, img in enumerate(imgsKpts):
    cv2.imshow(str(idx), img)
visualize_pcd_plt_4cam(pcd_four_cam)


print(ICP_with_timestamp(pcd_o3d, "seq1", local_timestamp[1], 0.1))
