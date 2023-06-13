import shutil
import os

dst_folder = "./ITRI_dataset"
src_folder = "./ITRI_DLC2"

seq_list = os.listdir(src_folder)

for seq in seq_list:

    if os.path.isfile(os.path.join(src_folder,seq)):
        continue
    # init pose
    src_parent_path = os.path.join(src_folder, seq, "new_init_pose")
    ts_folders = os.listdir(src_parent_path)
    for ts_folder in ts_folders:
        dst_path = os.path.join(dst_folder, seq, "dataset",  ts_folder, "initial_pose.csv")
        src_path = os.path.join(src_parent_path, ts_folder, "initial_pose.csv")
        shutil.copyfile(src_path, dst_path)
    
    # localization timestamp
    dst_path = os.path.join(dst_folder, seq, "localization_timestamp.txt")
    src_path = os.path.join(src_folder, seq, "localization_timestamp.txt")
    shutil.copyfile(src_path, dst_path)

    # gt_pose
    dst_path = os.path.join(dst_folder, seq, "gt_pose.txt")
    src_path = os.path.join(src_folder, seq, "gt_pose.txt")
    if os.path.exists(src_path):
        shutil.copyfile(src_path, dst_path)
