import cv2
import numpy as np
from Utility import Transform
from Utility import NormalizePoint, NormalizeImageIndex, NormalizeMatrix
from Utility import ext_cam

class VisualOdometry():
    """
    Visual Odometry Class

    General Idea: Read and store first frame,feature,descriptor -> Read N next frame to compute T(1,N), where the base frame 1 is the first frame.
    Algorithm: 
        Turn input frame to gray -> compute keyfeature -> compute descriptor for keyfeature - 
        > match the keyfeature between two frame -> compute pose from essential matrix -
        > compute T(1,N) with T(1,N-1)@T(N-1,N) -> store frame,feature,descriptor for next iteration

    Init_Attributes:
        _camera (Camera): This describe the intrinsic property of camera (init_Pose, resolution, camera_Matrix, distortion_coefficients, rectification_matrix, projection_matrix)
        _feature_detector: Any feature detector algorithm, as long as it satisfy this 
            API: feature_detector(grayImage,Mask) -> keyfeature: tuple[cv2.KeyPoint]
        _descriptors_computer: Any descriptors computer algorithm as long as 
            API: descriptors_computer(grayImage, keyfeature) -> keyfeature: tuple[cv2.KeyPoint], descriptor: numpy.ndarray[numpy.ndarray]
        _matcher: 
            API: matcher(keyfeature_frame_1,keyfeature_frame_2) -> tuple[cv2.DMatch]
        _findEssentialMat:
            API: findEssentialMat(p1,p2,Camera_Matrix) -> essential_mat, mask
        _isDebug: If is set to False, debug image would not be compute nor store (speedup)

    Attributes:
        public:
            Odom: Data structure for storing Odometry, it only have two attributes, pose_tf(homogenous pose) and timestamp
            debugFrame: ONLY USABLE IF isDebug == True, this stores two image for debugging
                debugFrame[0]: Image with Keypoint draw on Image
                debugFrame[1]: Image with Matched keypoint between current frame and previous frame.
        private:
            _tf: transformation class from Ultility.py, use for converting between euclidean and homogenous coordinate
            _frame_prev: store previous frame
            _keypoints_prev: store previous keyfeature
            _keypoints_curr: store current keyfeature
            _descriptors_prev: store previous keyfeature
            _descriptors_curr: store current keyfeature
            _normalize: normalization class from Ultility.py, not used yet, not sure if it is required.


    Methods:
        computeAndUpdateOdom(frame, time): Compute and update odometry, please use this before reading Odom.

    ToDo: Bundle Adjustment
    """
    def __init__(self, camera_object, feature_detector, descriptors_computer, matcher, findEssentialMat, isDebug=False):
            self.Odom = Odometry(camera_object.init_Pose,0)
            self._tf = Transform()
            self._camera = camera_object
            self._feature_detector = feature_detector
            self._descriptors_computer = descriptors_computer
            self._matcher = matcher
            self._findEssentialMat = findEssentialMat 

            self._frame_prev=None

            self._keypoints_prev=[]
            self._keypoints_curr=[]
            self._descriptors_prev=[]
            self._descriptors_curr=[]
            self._normalize = NormalizePoint()
            self._isDebug = isDebug
            if isDebug:
                self.debugFrame=[np.zeros((1,1)),np.zeros((1,1))]
            
            self.map = np.zeros((0,3))
            self.extrinsic_to_baseline = ext_cam.f_c.get_Transform_to_baselink()
            self.proj_prev = self._camera.camera_Matrix @ np.eye(3,4)
            self.proj_curr = self.proj_prev
            self.inlier_thr = 100
            self.timestamp_pcd_count = []
            self.pcd_color = np.zeros((0,3))
            self.route = np.zeros((0,3))
            self.map_tmp = None
            
            # variables for point adjustment and point adding
            self.prev_frame_keypnts_for_triangulate = None

    def computeAndUpdateOdom(self,frame,time):
        # Convert frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        self._keypoints_curr = self._feature_detector(gray,self._camera.mask)
        self._keypoints_curr, self._descriptors_curr = self._descriptors_computer(gray, self._keypoints_curr)

        if len(self._keypoints_prev) > 0:
            # self.count_frame += 1
            # Match keypoints and compute essential matrix
            matches = self._matcher(self._descriptors_prev, self._descriptors_curr)

            # temp = []
            # for m in matches:
            #     if m.distance < 30:
            #         temp.append(m)
            #     #else:
            #         #print(m.distance)
            # matches=temp
            # p1 = self._normalize.UndistorPoint(self._keypoints_prev,[m.queryIdx for m in matches],self._camera)
            # p2 = self._normalize.UndistorPoint(self._keypoints_curr,[m.trainIdx for m in matches],self._camera)
            
            # normalize keypoints from image index to [-1, 1]
            """Normalize"""
            keypoints_prev = np.float32([self._keypoints_prev[m.queryIdx].pt for m in matches])
            keypoints_curr = np.float32([self._keypoints_curr[m.trainIdx].pt for m in matches])

            essential_mat, mask = self._findEssentialMat(
                keypoints_curr,
                keypoints_prev,
                camera_Matrix=self._camera.camera_Matrix
            )
            inlier_num, R, t, inlier_mask = cv2.recoverPose(essential_mat,
                                        keypoints_curr,
                                        keypoints_prev,
                                        cameraMatrix=self._camera.camera_Matrix)
            
            inlier_mask = inlier_mask.squeeze().astype(bool)
            if inlier_num < len(keypoints_prev)*0.5:
                return
            # add inlier_num, if lower than a threshold, then dont use this frame
            """end"""
            

            # T(1,k) = T(1,n) @ T(n,k)
            self.Odom.pose_tf = (self.Odom.pose_tf) @ (self._tf.homogenousCoordinate(np.squeeze(t),R))
                   
            
            self.proj_curr = self.proj_prev @ (self._tf.homogenousCoordinate(np.squeeze(-t),R.T))

            tri_pnts = cv2.triangulatePoints(self.proj_prev, self.proj_curr, keypoints_prev[inlier_mask].T, keypoints_curr[inlier_mask].T)
            tri_pnts /= tri_pnts[3]
            
            # reproject and remove
            remove = np.zeros(inlier_num, dtype=bool)
            # reproject back
            reproj_curr = self.proj_curr @ tri_pnts
            reproj_prev = self.proj_prev @ tri_pnts
            # remove z<0
            remove[reproj_curr[2] < 0] = True
            
            # remove z<0
            remove[reproj_prev[2] < 0] = True

            reproj_curr /= reproj_curr[2]
            reproj_prev /= reproj_prev[2]

            error_dist_curr = reproj_curr[:2] - keypoints_curr[inlier_mask].T
            error_dist_prev = reproj_prev[:2] - keypoints_prev[inlier_mask].T
            error_dist_curr = (np.linalg.norm(error_dist_curr, 2, axis=0))
            error_dist_prev = (np.linalg.norm(error_dist_prev, 2, axis=0))
            # remove list
            reproj_error_threshold = 2
            remove[error_dist_curr > reproj_error_threshold] = True
            remove[error_dist_prev > reproj_error_threshold] = True
            
            # print("reproj curr:", error_mean_curr)
            # print("reproj prev:", error_mean_prev)
            
            
            # set condition: if the error over a threshold, then discard it?

            tri_pnts = tri_pnts[:3].T
            
            """ record car location """
            car_location, _ = self.Odom.getPoseByEuclidean()
            self.route = np.concatenate((self.route, car_location.reshape(1,3)), axis=0)


            """ record pcd """
            
            dist_tri_pnts = np.linalg.norm(tri_pnts- car_location.reshape(1,3), 2, axis=1)
            # midrange = (max(dist_tri_pnts)-min(dist_tri_pnts))/2
            # print("midrange of pcd:", midrange)
            # print("mean of z:", np.mean(tri_pnts[:,2]))
            remove[dist_tri_pnts > 25] = True
            

            # if error_mean_curr < 10 and error_mean_prev < 10 and midrange < 50:
                
            # print("produced: ",tri_pnts.shape[0])
            
            tri_pnts = tri_pnts[(1-remove).astype(bool)]

            # print("in range:", tri_pnts.shape[0])
            self.timestamp_pcd_count.append(tri_pnts.shape[0])
            self.map = np.concatenate((self.map, tri_pnts), axis=0)
                
                # point adding views
                # label with frame 1 2 3, 3 is the most current, and 1 is the most previous
                # keypnts now
            """
                curr_frame_keypts_3 = keypoints_curr[inlier_mask].T
                curr_frame_keypts_2 = keypoints_prev[inlier_mask].T
                # moment prev frame
                prev_frame_keypts_1 = self.prev_frame_keypnts_for_triangulate["prev"]
                prev_frame_keypts_2 = self.prev_frame_keypnts_for_triangulate["curr"]
                
                # point adjustment
                # duplicated points calculated
                curr_frame_duplicate_array = np.zeros(curr_frame_keypts_2.shape[0], dtype=bool)
                prev_frame_duplicate_array = np.zeros(prev_frame_keypts_2.shape[0], dtype=bool)

                for i in range(curr_frame_keypts_2.shape[0]):
                    row_in_curr = curr_frame_keypts_2[i]
                    x_prev_exist = np.isin(prev_frame_keypts_2[:,0], row_in_curr[0])
                    y_prev_exist = np.isin(prev_frame_keypts_2[:,1], row_in_curr[1])
                    pnts_prev_exist = x_prev_exist & y_prev_exist
                    index = np.where(pnts_prev_exist)[0]
                    if index.size != 0:
                        curr_frame_duplicate_array[i] = True
                        prev_frame_duplicate_array[index[0]] = True
                
                curr_frame_duplicate_keypts_3 = curr_frame_keypts_3[curr_frame_duplicate_array]
                prev_frame_duplicate_keypts_1 = prev_frame_keypts_1[prev_frame_duplicate_array]
                    
                # point adding
                """
            
            if self._isDebug:
                #Draw Keypoint only on current image
                inlier_mask = list(np.where(inlier_mask == True)[0])
                keypnts_prev_drawn = [self._keypoints_prev[i] for i in inlier_mask]
                keypnts_curr_drawn = [self._keypoints_curr[i] for i in inlier_mask]
                # matches = [matches[i] for i in inlier_mask]
                self.debugFrame[0] = cv2.drawKeypoints(frame, keypnts_curr_drawn, None, color=(0,255,0), flags=0)
                #Draw matching point between current and previous image
                self.debugFrame[1] = cv2.drawMatches(self._frame_prev, self._keypoints_prev, frame, self._keypoints_curr, matches, None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
                
        self._frame_prev = frame

        # point adjustment variables update
        # self.prev_frame_keypnts_for_triangulate = {"prev_pnt": self._keypoints_prev, "curr_pnt": self._keypoints_curr, "prev_proj": self.proj_prev}

        self._keypoints_prev = self._keypoints_curr
        self._descriptors_prev = self._descriptors_curr

        self.proj_prev = self.proj_curr

    def o3dcolor(self):
        all_frame_count = len(self.timestamp_pcd_count)
        color_linspace = np.linspace(0, 255, all_frame_count).astype(np.uint8)
        for i in range(all_frame_count):
            color_i = np.array([color_linspace[i], 255-color_linspace[i], 0]).reshape(1,3)
            self.pcd_color = np.concatenate((self.pcd_color, np.repeat(color_i, self.timestamp_pcd_count[i], axis=0)))
        return self.pcd_color

class Odometry():
    def __init__ (self,pose_tf,timestamp):#,speed,angular_speed,timestamp):
        self.pose_tf = pose_tf
        self.timestamp = timestamp
    #    self.speed = speed
    #    self.angular_speed = angular_speed
    def getPoseByEuclidean(self):
        return Transform().EuclideanCoordinate(self.pose_tf)
    def getPoseByHomogenous(self):
        return self.pose_tf



