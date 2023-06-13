from dataset2numpy import Dataset
from argparse import ArgumentParser
import numpy as np
import os
from ICP import ICP_with_timestamp

parser = ArgumentParser()
parser.add_argument("--seq", type=str, default="seq1")
args = parser.parse_args()

seq = args.seq
print(seq)
pred_pose_filename = f"pred_pose_{seq}.txt"
dataset_path = "../../ITRI_dataset/"
dataset = Dataset(seq, loc=dataset_path)
local_timestamp = dataset.get_time_stamp(with_GT=True)

thresholds = [0.11, 0.13, 0.15, 0.17, 0.19, 0.21, 0.25, 0.3]

for th in thresholds:
    pred_pose = np.zeros((0,2))
    writeFile = True
    for lts in local_timestamp:
        pcd = np.loadtxt(os.path.join("pcd",seq,lts+".csv"), dtype=np.float32, delimiter=",")
        pred_pose_ts, no_correspondence = ICP_with_timestamp(pcd, seq, lts, th)
        if no_correspondence: 
            writeFile = False
            break
        pred_pose = np.vstack((pred_pose, np.array(pred_pose_ts)))
    
    if writeFile:
        print("seq:", seq)
        print("threshold:",th)
        np.savetxt(pred_pose_filename, pred_pose, delimiter=" ")
        break

