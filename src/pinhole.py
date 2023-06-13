import cv2
import matplotlib.pyplot as plt
import numpy as np

import config
from dataset2numpy import Dataset
from RoadMarker import RoadMarkerDetection
from Utility import Transform, ext_cam


class Pinhole:
    def __init__(self, camera_object, ext_cam, zw=-1.63):
        self._tf = Transform()
        self._camera = camera_object
        self.to_baselink = ext_cam.get_Transform_to_baselink()
        self.from_baselink = ext_cam.get_Transform_from_baselink()
        self.extrinsic = np.eye(3, 4)
        self.zw = zw  # unit: mm
        self.intrinsic = self._camera.projection_matrix[:3,:3]
        self.intrinsic_inv = np.linalg.inv(self.intrinsic)
        self.Rt = np.eye(3,4)
        self.update_Rt_pinv()

        # get the plane of ground
        self.g_dot = (self.from_baselink @ np.array([0, 0, self.zw, 1]))
        self.g_dot = self.g_dot[:3] / self.g_dot[3]
        # print("gdot", self.g_dot)
        self.normal = (self.from_baselink[:3,:3] @ np.array( [0, 0, 1]))
        # self.normal = self.normal[:3] / self.normal[3]
        # print(self.g_dot)
        # print("normal",self.normal)
        self.fx = self.intrinsic[0,0]
        self.fy = self.intrinsic[1,1]
        self.cx = self.intrinsic[0,2]
        self.cy = self.intrinsic[1,2]

    def reconstruct_from_kpts_plane(self, kpts):
        if kpts.ndim == 1:
            kpts = kpts.reshape(2, 1)

        if kpts.shape[1] == 2:
            kpts = kpts.T
        
        XYZc = self.get_kp_vector_camera_coordinate(kpts)
        scale = (self.normal @ self.g_dot)/ (self.normal @ XYZc)
        # print(scale)
        scale = scale.reshape(1, -1).repeat(3, axis=0)
        XYZc = scale * XYZc
        
        

        XYZc = np.vstack((XYZc,np.ones(XYZc.shape[1])))
        world_kpts = self.to_baselink @ XYZc

        w = world_kpts[3]
        w[w == 0] = 1
        world_kpts /= world_kpts[3]

        
        return self.filter_points_within_distance(world_kpts[:3].T)
    
    def filter_points_within_distance(self, world_coordinates, max_distance=30):
        # Compute the Euclidean distance from origin.
        dist_from_origin = np.sqrt(np.sum(world_coordinates**2, axis=1))

        # Print out the number of points that are further than max_distance.
        # print("Points farther than {}m: ".format(max_distance), world_coordinates[dist_from_origin > max_distance].shape[0])

        # Filter out points that are further than max_distance from the origin.
        world_coordinates = world_coordinates[dist_from_origin <= max_distance]

        return world_coordinates
    
    def get_kp_vector_camera_coordinate(self, kpts):
        # input:  (2,N) array
        # output: (3,N) array
        if kpts.shape[1] == 2:
            kpts = kpts.T
        
        XYZc = np.ones((3, kpts.shape[1]))
        XYZc[0] = (kpts[0] - self.cx) / self.fx
        XYZc[1] = (kpts[1] - self.cy) / self.fy

        return XYZc
        

    def reconstruct_from_kpts(self, kpts):
        # shape of kp: (N,2)
        if kpts.shape[1] == 2:
            kpts = kpts.T

        kpts = np.vstack((kpts,np.ones(kpts.shape[1])))
        world_kpts = self.to_baselink @ self.Rt_pinv @ self.intrinsic_inv @ kpts
        
        w = world_kpts[3]
        w[w == 0] = 1
        world_kpts /= world_kpts[3]
        scale = ((self.zw / world_kpts[2]).reshape(1, -1)).repeat(3, axis=0)
        
        world_kpts = (world_kpts[:3] * scale).T
        return world_kpts
    
    def update_Rt_pinv(self):
        self.Rt_pinv = np.linalg.pinv(self.Rt)
        return self.Rt_pinv
    
    def update_Rt(self, R, t):
        Rt_0 = np.vstack((self.Rt, np.array([[0, 0, 0, 1]])))
        Rt_1 = self._tf.homogenousCoordinate(np.squeeze(t), R)
        self.Rt = Rt_0 @ Rt_1
        return self.Rt

def visualize_pcd_plt(pcd):
    # ignore z
    plt.scatter(pcd[:,0],pcd[:,1], s=0.5)
    plt.show()  

def visualize_pcd_plt_4cam(pcds):
    plt.gca().set_aspect('equal')
    for pcd in pcds:
        p = pcd["pcd"]
        c = pcd["color"]
        plt.scatter(p[:,0], p[:,1], s=1, c=c)
    plt.show()
    

if __name__ == "__main__":
    front_cam = Pinhole(config.Front_Cam, ext_cam.f_c)
    RMD = RoadMarkerDetection()

    dataset = Dataset("seq1", loc="../../ITRI_dataset/")
    all_timestamp = dataset.get_time_stamp()
    local_timestamp = dataset.get_time_stamp(with_GT=True)
    direction = dataset.get_camera_dir()
    raw_image_front = dataset.get_raw_image([all_timestamp[i] for i in np.where(direction == 0)[0]])
    frame_num, _, _, _ = raw_image_front.shape
    road_marker_list = dataset.get_detect_road_marker()
    front_road_marker_list = [road_marker_list[i] for i in np.where(direction == 0)[0]]
    front_time_stamp = [all_timestamp[i] for i in np.where(direction == 0)[0]]
    
    mask = front_cam._camera.mask
    frame = raw_image_front[0]
    kp_road, imgKpts, _ = RMD.detection(frame.copy(), mask, front_road_marker_list[0], config)
    imgKpts = cv2.resize(imgKpts, (640, 640))

    kp_road = np.float32([kp.pt for kp in kp_road])
    # print(kp_road)
    pcd = front_cam.reconstruct_from_kpts(kp_road)
    visualize_pcd_plt(pcd)

