import numpy as np
from numpy import asarray
import csv
import os
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from tqdm import tqdm
import pandas as pd
import cv2

class Dataset:
    def __init__(self, name="seq1", loc=".", img_npz_loc=""):
        self.name = name
        self.all_timestamp = []
        self.gt_timestamp = []
        self.loc = os.path.join(loc, name)
        self.image = None
        if img_npz_loc != "":
            print("load raw_image...")
            with open(img_npz_loc,'rb') as f:
                self.image = np.load(f)['arr_0']
                # self.image = tmp['arr_0']
            print("end of loading images")
        self.order = None
    def get_raw_image(self, save_loc="", silence=False):
        if self.image is not None:
            return self.image
        else: 
            self.image = []
        save_loc = f"./raw_images_{self.name}.npz" if save_loc=="" else save_loc 
        if not self.check_do_time_stamp():
            self.get_time_stamp()
        for ts in tqdm(self.all_timestamp, desc="load images", disable=silence):
            filename = os.path.join(self.loc, "dataset", ts, "raw_image.jpg")
            img = asarray(cv2.imread(filename))
            self.image.append(img)
        self.image = asarray(self.image)
        print("\nsave to npz...")
        with open(save_loc, "wb") as f:
            np.savez_compressed(f, self.image)
        return self.image

    def get_raw_image(self, timestamp=[], silence=False):
        imgs = []
        for ts in tqdm(timestamp, desc="get images...", disable=silence):
            filename = os.path.join(self.loc,"dataset", ts, "raw_image.jpg")
            img = asarray(cv2.imread(filename))
            imgs.append(img)
        return asarray(imgs)

    def get_time_stamp(self, with_GT= False):       # done
        # localization_timestamp repeated at line 620, that is, only 619 timestamp is needed
        filename="localization_timestamp.txt" if with_GT else "all_timestamp.txt"
        file_path = os.path.join(self.loc, filename)
        f = open(file_path)
        lines = f.readlines()
        lines = [line.rstrip("\n") for line in lines]
        if not with_GT:
            self.all_timestamp = lines
        else:
            self.gt_timestamp = lines
        f.close()
        return self.gt_timestamp if with_GT else self.all_timestamp
        

    def get_camera_dir(self):   
        # four directions: f, fl, fr, b -> encode to 0, 1, 2, 3
        if not self.check_do_time_stamp():
            self.get_time_stamp()
        
        dir_dict = {"f":0,"fl":1,"fr":2,"b":3}
        self.camera_dir = np.zeros(len(self.all_timestamp))
        for i, ts in  tqdm(enumerate(self.all_timestamp), desc="get camera dir"):
            filename = os.path.join(self.loc, "dataset", ts, "camera.csv")
            with open(filename, "r") as f:
                row = f.readline()
                dir_info = row.split("/")[-1]
                dir = dir_info.split("_")[-2]
                self.camera_dir[i] = dir_dict[dir]
        return self.camera_dir

    def get_camera_dir(self, ts):
        oneTs = False
        if type(ts) == str:
            ts = [ts]
            oneTs = True
        dir_dict = {"f":0,"fl":1,"fr":2,"b":3}
        dir_list = []
        for i in ts:
            filename = os.path.join(self.loc,"dataset", i, "camera.csv")
            with open(filename, "r") as f:
                row = f.readline()
                dir_info = row.split("/")[-1]
                dir = dir_info.split("_")[-2]
                dir_list.append(dir_dict[dir])
        return dir_list[0] if oneTs else dir_list 
        
        
    def get_detect_road_marker(self, silence=False):       
        # store with timestamp
        # element: [x1, y1, x2, y2, class, prob]
        # class: 0:zebracross, 1:stopline, 2:arrow, 3:junctionbox, 4:other
        self.detect_road_marker_list = []
        for ts in tqdm(self.all_timestamp, desc="get detect marker",disable=silence):
            filename = os.path.join(self.loc, "dataset", ts, "detect_road_marker.csv")
            if os.path.getsize(filename):
                road_marker = pd.read_csv(filename)
                road_marker.columns = ['x1', 'y1', 'x2', 'y2', 'class_id', 'probability']
            else:
                road_marker = []
            self.detect_road_marker_list.append(road_marker)
            
        return self.detect_road_marker_list

    def get_detect_road_marker(self, timestamp):
        if type(timestamp) == str:
            timestamp = [timestamp]
        road_markers = []
        for ts in timestamp:
            filename = os.path.join(self.loc, "dataset", ts, "detect_road_marker.csv")
            if os.path.getsize(filename):
                road_marker = pd.read_csv(filename)
                road_marker.columns = ['x1','y1', 'x2', 'y2', 'class_id', 'probability']
            else:
                road_marker = []
            road_markers.append(road_marker)
        return road_markers

    def get_initial_pose(self):
        # camera pose, (x, y, z, angle)
        self.initial_pose = []
        for ts in tqdm(self.all_timestamp, desc="get initial pose"):
            filename = os.path.join(self.loc, "dataset", ts, "initial_pose.csv")
            pos_camera = np.loadtxt(filename, dtype=float, delimiter=",")
            self.initial_pose.append(pos_camera)
        self.initial_pose = asarray(self.initial_pose)
        return self.initial_pose
    
    def get_sub_map(self):
        # all sub map are the same in each timestamp
        self.sub_map_list = []
        for ts in tqdm(self.all_timestamp, desc="get sub_map"):
            filename = os.path.join(self.loc, "dataset", ts, "sub_map.csv")
            self.sub_map = np.loadtxt(filename, dtype=float, delimiter=",")
            self.sub_map_list.append(self.sub_map)
        return self.sub_map_list
        
    
    def check_do_time_stamp(self):
        return False if self.all_timestamp == [] else True

    def show_img_contour(self, timestamp= None):
        # timestamp = self.all_timestamp[0]
        color = {0: "r", 1: "g", 2: "b", 3:"y", 4:"black"}
        fig, ax = plt.subplots()
        imgname = os.path.join(self.loc, "dataset", timestamp, "raw_image.jpg")
        img = asarray(cv2.imread(imgname))
        markername = os.path.join(self.loc, "dataset", timestamp, "detect_road_marker.csv")
        ax.imshow(img)
        marker = np.loadtxt(markername, dtype=float, delimiter=",")
        
        for row in marker:
            edgecolor = color[int(row[4])]
            rect = Rectangle((row[0], row[1]), row[2]-row[0], row[3]-row[1], linewidth=1, edgecolor=edgecolor, facecolor="none")
            ax.add_patch(rect)
        plt.show()

    def get_gt_pose(self):
        filename = os.path.join(self.loc, "gt_pose.txt")
        self.gt_pose = None
        if os.path.exists(filename):
            self.gt_pose = np.loadtxt(filename, dtype=float, delimiter=" ")
        return self.gt_pose
    
    def get_raw_speed(self):
        # the timestamp in other_data is different from dataset
        self.raw_speed = []
        timestamp = self.get_raw_speed_timestamp()
        for ts in tqdm(timestamp, desc="get raw_speed"):
            self.raw_speed.append(np.loadtxt(os.path.join(self.loc,"other_data", ts+"_raw_speed.csv"), dtype=float))
        self.raw_speed = asarray(self.raw_speed)
        return self.raw_speed

        

    def get_raw_imu(self):
        # first row: 4 item
        # second:    3 item
        # third:     3 item
        # [1:,-1] is nonsence
        self.raw_imu = []
        timestamp = self.get_raw_imu_timestamp()        
        for ts in tqdm(timestamp, desc="get raw_imu"):
            raw_imu = np.zeros((3,4))
            with open(os.path.join(self.loc,"other_data", ts+"_raw_imu.csv"),"r") as f:
                reader = csv.reader(f, quoting=csv.QUOTE_NONNUMERIC)
                for i, row in enumerate(reader):
                    raw_imu[i] = row[i]
                
            self.raw_imu.append(raw_imu)
        self.raw_imu = asarray(self.raw_imu)
        return self.raw_imu

    def get_raw_speed_timestamp(self):
        filename = os.path.join(self.loc, "other_data", "raw_speed_time_stamp.txt")
        if os.path.exists(filename):
            return np.loadtxt(filename, dtype=str)
        else:
            f = open(filename, "w")
            for file in tqdm(os.listdir(os.path.join(self.loc, "other_data")),desc="get raw_imu"):
                if file.endswith("_raw_speed.csv"):
                    f.write(file[:-14])
                    f.write('\n')
            f.close()
        return np.loadtxt(filename, dtype=str)

    def get_raw_imu_timestamp(self):
        filename = os.path.join(self.loc, "other_data", "raw_imu_time_stamp.txt")
        if os.path.exists(filename):
            return np.loadtxt(filename, dtype=str)
        else:
            f = open(filename, "w")
            for file in tqdm(os.listdir(os.path.join(self.loc, "other_data")),desc="get raw_imu"):
                if file.endswith("_raw_imu.csv"):
                    f.write(file[:-12])
                    f.write('\n')
            f.close()
        return np.loadtxt(filename, dtype=str)

    def timestamp2num(self, timestamp):
        timestamp_num = []
        for ts in timestamp:
            ts = ts.split("_")
            if len(ts[1]) < 9:
                ts[1] += "0"
            ts_num = int(ts[0][-5:] + ts[1])
            timestamp_num.append(ts_num)
            
        return np.array(timestamp_num)

    def get_4_cam_at_ts(self, timestamp):
        if self.order is None:
            self.get_camera_order()
        index = self.all_timestamp.index(timestamp)
        dir = self.get_camera_dir(self.all_timestamp[index])
        if dir == self.order[0]:
            return self.all_timestamp[index:index+4]
        elif dir == self.order[1]:
            return self.all_timestamp[index-1:index+3]
        elif dir == self.order[2]:
            return self.all_timestamp[index-2:index+2]
        else:
            return self.all_timestamp[index-3:index+1]

    def get_camera_order(self):
        self.order = self.get_camera_dir(self.all_timestamp[:4])
        
        pass

if __name__ == "__main__":
    d = Dataset(name="seq1", img_npz_loc="./raw_images_seq1.npz")
    d.get_time_stamp()
    # d.get_raw_image()
    # d.get_gt_pose()
    print(d.get_raw_image())
    # print(im['arr_0'])
    
    
    