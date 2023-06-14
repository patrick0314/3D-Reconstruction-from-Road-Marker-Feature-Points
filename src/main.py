import config
from dataset2numpy import Dataset
import numpy as np
from Utility import ext_cam
from RoadMarker import RoadMarkerDetection
from pinhole import Pinhole
from ICP import ICP_with_timestamp
from eval import benchmark
from tqdm import tqdm
from argparse import ArgumentParser
import os


parser = ArgumentParser()
parser.add_argument("--seq", type=str, default="seq1")
parser.add_argument("--silence","-s", type=int, default=0, help="0 for False, others for True")
parser.add_argument("--ICP_threshold", type=float, default=0.2)
parser.add_argument("--coefficient", type=float, default=0.01)
parser.add_argument("--ratio","-r", type=float, default=18)
parser.add_argument("--lowerbound", "-l", type=int, default=130)
parser.add_argument("--max_dist","-d", type=int, default=30)
args = parser.parse_args()

silence = (args.silence != 0)
seq = args.seq
pred_pose_filename = f"pred_pose_{seq}.txt"
dataset_path = config.dataset_path
dataset = Dataset(seq, loc=dataset_path)
all_timestamp = dataset.get_time_stamp()
local_timestamp = dataset.get_time_stamp(with_GT=True)

# camera
f_c = Pinhole(config.Front_Cam, ext_cam.f_c, max_dist=args.max_dist)
f_r = Pinhole(config.Right_Cam, ext_cam.f_r, max_dist=args.max_dist)
f_l = Pinhole(config.Left_Cam,  ext_cam.f_l, max_dist=args.max_dist)
f_b = Pinhole(config.Back_Cam,  ext_cam.b,   max_dist=args.max_dist)

cameras = {0:f_c, 1:f_l, 2:f_r, 3:f_b}
masks = {0: f_c._camera.mask, 1: f_l._camera.mask, 2: f_r._camera.mask, 3: f_b._camera.mask}
colors = {0: "red", 1:"green",2:"blue", 3:"black"}

# road markers
RMD = RoadMarkerDetection(args.coefficient, args.ratio, args.lowerbound)
print('### coefficient = {} ###'.format(args.coefficient))
print('### ratio = {} ###'.format(args.ratio))
print('### lowerbound = {} ###'.format(args.lowerbound))
print('### max_dist = {} ###'.format(args.max_dist))

save_path = os.path.join(config.output_path, seq)
if not os.path.isdir(save_path):
    os.makedirs(save_path)


no_corr_count = 0
pred_pose = np.zeros((0,2))
require_4_cam_ts = []
ICP_threshold = args.ICP_threshold
for i in tqdm(local_timestamp, desc="gen pcd...", disable=silence):
    if i not in require_4_cam_ts:
        require_4_cam_ts = dataset.get_4_cam_at_ts(i)
        raw_images = dataset.get_raw_image(require_4_cam_ts, silence=True)
        road_marker_list = dataset.get_detect_road_marker(require_4_cam_ts)
        direction = dataset.get_camera_dir(require_4_cam_ts)
        pcd_o3d = np.zeros((0,3))
        for j in range(4):
            camera = cameras[direction[j]]
            mask = masks[direction[j]]
            frame = raw_images[j]
            road_marker = road_marker_list[j]
            if len(road_marker) == 0:
                #print("\nNo roadmarker!!", require_4_cam_ts[j])
                continue
            kp_road, imgKeypoint, imgContour = RMD.detection(frame.copy(), mask, road_marker, config, direction[j])
            kp_road = np.float32([kp.pt for kp in kp_road])
            if kp_road.size == 0:
                #print("\nNo kp!!", require_4_cam_ts[j])
                continue
            pcd = camera.reconstruct_from_kpts_plane(kp_road)
            pcd_o3d = np.vstack((pcd_o3d, pcd))
    np.savetxt(f"{save_path}/{i}.csv", pcd_o3d, delimiter=",")
    pred_pose_ts, no_correspondence = ICP_with_timestamp(pcd_o3d, seq, i, ICP_threshold)
    if no_correspondence: no_corr_count+=1
    pred_pose = np.vstack((pred_pose, np.array(pred_pose_ts)))

np.savetxt(pred_pose_filename, pred_pose, delimiter=" ")
print("no corr count:",no_corr_count)

if seq[:3] == "seq":
    benchmark(dataset_path, seq, src_path=pred_pose_filename)

