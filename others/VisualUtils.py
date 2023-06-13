import cv2
import numpy as np
from Utility import Transform
from Utility import NormalizePoint, NormalizeImageIndex, NormalizeMatrix
from Utility import ext_cam

class ShortTimeReconstruct():
    """ use about 15 frame to compute """
    def __init__(self, camera_object, feature_detector, descriptors_computer, matcher, findEssentialMat):
        self._tf = Transform()
        self._camera = camera_object
        self._feature_detector = feature_detector
        self._descriptors_computer = descriptors_computer
        self.matcher = matcher
        self._findEssentialMat= findEssentialMat
        
    def setCenterImage(self, centerImage, mask):
        gray = cv2.cvtColor(centerImage, cv2.COLOR_RGB2GRAY)
        self._keypoints_center = self._feature_detector(gray, mask)
        self._keypoints_center, self._descriptors_center = self._descriptors_computer(gray, self._keypoints_center)

    def motionsFromFrame(self, frame, camera_matrix, p_center):
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        self._keypoints_curr = self._feature_detector(gray)
        self._keypoints_curr, self._descriptors_curr = self._descriptors_computer(gray, self._keypoints_curr)

        # match with center frame
        matches = self.matcher(self._descriptors_center, self._descriptors_curr)
        keypoints_center = np.float32([self._keypoints_center[m.queryIdx].pt for m in matches])
        keypoints_curr = np.float32([self._keypoints_curr[m.trainIdx].pt for m in matches])

        # trim keypoint?
        # keypoints_center, keypoints_curr = self.trim_keypoints(keypoints_center, keypoints_curr, threshold=100)
        
        if keypoints_center.size == 0:
            print("No match points!")
            return 
        
        # get essential_matrix and motions 
        essential_mat, mask = self._findEssentialMat(
            keypoints_center,
            keypoints_curr,
            camera_Matrix = camera_matrix
        )

        inlier_num, R, t, inlier_mask = cv2.recoverPose(
            essential_mat,
            keypoints_center,
            keypoints_curr,
            cameraMatrix = camera_matrix 
        )

        inlier_mask = inlier_mask.squeeze().astype(bool)

        proj_curr = p_center @ self._tf.homogenousCoordinate(np.squeeze(t), R)
        
        return (proj_curr,  
                keypoints_center[inlier_mask],      
                keypoints_curr[inlier_mask])


    def trim_keypoints(self, keypoints_center, keypoints_curr, threshold):
        # the pixel location wont get too much displacement?
        # set maximum displacement is 100 pixel
        distance = np.linalg.norm(keypoints_center - keypoints_curr, 2, axis=1)
        
        return keypoints_center[distance < threshold], keypoints_curr[distance < threshold]
    
    def reconstruct(self, keypts_center, keypts_curr, P_center, P_curr):
        # print("keypnts center", keypts_center.shape)
        # print("keypnts curr",keypts_curr.shape)
        if keypts_center.ndim == 1:
            keypts_center = keypts_center.reshape(-1, 1)
            keypts_curr = keypts_curr.reshape(-1, 1)


        keypts_center = keypts_center.T # to shape(2, N)
        keypts_curr = keypts_curr.T     # to shape(2, N)
        tri_pnts = cv2.triangulatePoints(P_center, P_curr, keypts_center, keypts_curr)
        tri_pnts /= tri_pnts[3]


        # reproject
        reproj_center = P_center @ tri_pnts
        reproj_curr = P_curr @ tri_pnts

        # remove some outliers
        remove = np.zeros(keypts_center.shape[1], dtype=bool)
        remove[reproj_center[2] <= 0] = True
        remove[reproj_curr[2] <= 0] = True
        
            # remove the point that is behind the camera
        reproj_center /= reproj_center[2]
        reproj_curr   /= reproj_curr[2]
        
        # print("center",reproj_center[:2])
        # print("curr", reproj_curr[:2])
            # set a threshold, if reproject error over this threshold, then remove
        reporj_error_th = 30
        error_dist_curr = reproj_curr[:2] - keypts_curr
        error_dist_center = reproj_center[:2] - keypts_center
        error_dist_curr = np.linalg.norm(error_dist_curr, 2, axis=0)
        error_dist_center = np.linalg.norm(error_dist_center, 2, axis=0)
        
        # print("error curr", error_dist_curr)
        # print("error center", error_dist_center)

        remove[error_dist_center > reporj_error_th] = True
        remove[error_dist_curr > reporj_error_th] = True

        tri_pnts = tri_pnts[:3].T

            # remove the pcd that out of 25m
        dist_tri_pnts = np.linalg.norm(tri_pnts, 2, axis=1)
        remove[dist_tri_pnts > 25] = True

        return tri_pnts[(1-remove).astype(bool)]

    def setCenterKptsnDes(self, kpts, des):
        self.centerKpts = kpts
        self.centerDes = des
    
    def motionsFromKptsnDes(self, kpts, des, camera_matrix, p_center):
        matches = self.matcher(self.centerDes, des)
        
        # good matches classified by dist
        kpts_center = np.float32([self.centerKpts[m.queryIdx].pt for m in matches])
        kpts_curr = np.float32([kpts[m.trainIdx].pt for m in matches])

        if kpts_center.size == 0:
            print("No match points")
            return
        
        essential_mat, mask = self._findEssentialMat(
            kpts_center,
            kpts_curr,
            camera_Matrix = camera_matrix
        )

        if mask is not None:
            kpts_center = kpts_center[mask.ravel() == 1]
            kpts_curr = kpts_curr[mask.ravel() == 1]
        else:
            print(essential_mat)
        P2s = get_P_from_E(essential_mat)

        ind = -1 
        kpts_center = np.vstack((kpts_center, np.ones((1, kpts_center.shape[1]))))
        kpts_curr = np.vstack((kpts_curr, np.ones((1, kpts_curr.shape[1]))))

        for i, P2 in enumerate(P2s):
            d1 = reconstruct_one_point(kpts_center[:,0], kpts_center[:,0], p_center, P2)
            P2_homo = np.linalg.inv(np.vstack([P2, [0,0,0,1]]))
            d2 = P2_homo[:3,:4] @ d1
            if d1[2] > 0 and d2[2] > 0:
                ind = i
        P2 = np.linalg.inv(np.vstack([P2s[ind], [0,0,0,1]]))[:3, :4]

        # inlier_mask = inlier_mask.squeeze().astype(bool)

        return ( P2,
                kpts_center.squeeze(),
                kpts_curr.squeeze(),
                matches
        )

    
def get_P_from_E(E):
	U, S, V = np.linalg.svd(E)

	if np.linalg.det(np.dot(U, V)) < 0:
		V = -V

	W = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
	P2s = [
		np.vstack((np.dot(U, np.dot(W, V)).T, U[:, 2])).T,
		np.vstack((np.dot(U, np.dot(W, V)).T, -U[:, 2])).T,
		np.vstack((np.dot(U, np.dot(W.T, V)).T, U[:, 2])).T,
		np.vstack((np.dot(U, np.dot(W.T, V)).T, -U[:, 2])).T]

	return P2s

def skew(x):
	return np.array([
		[0, -x[2], x[1]],
		[x[2], 0, -x[0]],
		[-x[1], x[0], 0]])


def reconstruct_one_point(pt1, pt2, m1, m2):
	A = np.vstack([
		np.dot(skew(pt1), m1),
		np.dot(skew(pt2), m2)])

	U, S, V = np.linalg.svd(A)
	P = np.ravel(V[-1, :4])

	return P / P[3]


def kp2np(kp_center, kp_curr, matches, threshold=0.8):
    matches = sorted(matches, key = lambda x:x.distance)
    kp_center = np.float32([kp_center[m.queryIdx].pt for m in matches])
    kp_curr = np.float32([kp_curr[m.trainIdx].pt for m in matches])
    
    return kp_center, kp_curr, matches