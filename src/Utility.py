import numpy as np
import cv2

def NormalizeMatrix(h, w):
    """
        give a matrix to normalize the index of pixel to [-1~1, -1~1], to let index (0,0) at the image center
        
        (0,0) ----------- (w,0)             (-1,-1) ----------- (1,-1)
          |                 |                  |                   |
          |                 |       ==>        |                   |
          |                 |                  |                   |
        (0,h) ----------- (w,h)             ( -1,1) ----------- ( 1,1)
    """

    normalizeMatrix = np.eye(3)

    # normalizeMatrix[:2, 2] = -1
    normalizeMatrix[2, 0] = -w/2
    normalizeMatrix[2, 1] = -h/2
    # normalizeMatrix = np.eye(3)
    # normalizeMatrix[0,0] = 2/w
    # normalizeMatrix[1,1] = 2/h
    # normalizeMatrix[2,:2] = -1
    return normalizeMatrix

def NormalizeImageIndex(h, w, keypoints, matrix=None):
    # x,y
    assert keypoints.shape[1] == 2 
    if matrix is None:
        normalizeMatrix = NormalizeMatrix(h=h, w=w)
    else: normalizeMatrix = matrix
    normalized_keypnts = np.concatenate((keypoints.T, np.array([np.ones(keypoints.shape[0])])), axis=0)
    normalized_keypnts = normalizeMatrix @ normalized_keypnts
    return (normalized_keypnts[:2]).T

def deNormalizeImageIndex(h, w, keypoints):
    assert keypoints.shape[1] == 2
    denormalized_keypnts = np.concatenate((keypoints.T, np.ones(keypoints.shape[0])), axis=0)
    denormalizeMatrix = np.linalg.inv(NormalizeMatrix(h=h, w=w))

    denormalized_keypnts = denormalizeMatrix @ denormalized_keypnts
    return (denormalized_keypnts[:2]).T
        
class Transform():
    def __init__(self):
        pass
    def homogenousCoordinate(self,Translation,Rotation):
        homogeneous_matrix = np.eye(4)
        homogeneous_matrix[:3, 3] = Translation
        homogeneous_matrix[:3, :3] = Rotation
        return homogeneous_matrix
    def EuclideanCoordinate(self,homogeneous_matrix):
        position = homogeneous_matrix[:3, 3]
        rotation = homogeneous_matrix[:3, :3]
        return position,rotation

class NormalizePoint():
    def __init__(self):
        pass
    def UndistorPoint(self,points,index,Camera):
        return [np.squeeze(cv2.undistortPoints(points[i].pt,Camera.camera_Matrix,Camera.distortion_coefficients)) for i in index]

class Camera():
    def __init__(self,yaml=None,init_Pose=None,resolution=None,camera_Matrix=None,distortion_coefficients=None,rectification_matrix=None,projection_matrix=None,mask=None):
        if yaml==None:
            self.init_Pose = np.array(init_Pose)
            self.resolution = np.array(resolution)
            self.camera_Matrix = np.array(camera_Matrix)
            self.distortion_coefficients = np.array(distortion_coefficients)
            self.rectification_matrix = np.array(rectification_matrix)
            self.projection_matrix = np.array(projection_matrix)
        else:
            """ToDo YAML file reading"""
            pass
        self.mask = mask

class Extrinsic_Camera:
    """
        This class is to record thej extrinsic matrix of camera, and provide 
        a matrix that transform back to baselink.
        For example, the origin of camera f_r is [0,0,0] in f_r coordinate(aka, the f_r's coordinate),
        we can transform it back to baselink by using the transform this class provide. 

        usage:
            __init__(arg_array=None, parent_Transform=None)
                arg_array: the argument that at file "camera_extrinsic_static_tf.launch".
                           total is 7 argments in a np array. 
                           Each is x, y, z, qx, qy, qz, qw respectively.
                           x, y, z are for translation. 
                           qx, qy, qz, qw are for rotation.
                parent_Transform: the matrix that transform parent coordinate to baselink
                                  the shape is (4, 4) 
            
    """
    def __init__(self, arg_array = None, parent_Transform=None):
        # arguments with default values
        self.Transform = np.eye(4)
        self.Translation = np.zeros(3)
        self.Q = np.zeros(4)
        self.pT = np.eye(4) # parent transform
        self.Transform_to_baselink = np.eye(4)
        self.Transform_from_baselink = np.eye(4)
        # if there are input arguments
        if arg_array is not None:
            assert arg_array.size==7
            self.Translation = arg_array[:3]
            self.Q = arg_array[3:]
        if parent_Transform is not None:
            assert parent_Transform.shape == (4,4)
            self.pT = parent_Transform
        self.get_Transform_to_parent_axis()
        self.get_Transform_to_baselink()
        self.get_Transform_from_baselink()
        self.z = (self.get_Transform_to_baselink() @ np.array([0,0,0,1]))[2]
    def get_Transform_to_parent_axis(self):
        """
            Rotation M calculated by qx, qy, qz, qw:
                [ 
                  1-s*(qy^2+qz^2)  s*(qx*qy-qw*qz)  s*(qx*qz+qw*qy)  0
                  s*(qx*qy+qw*qz)  1-s*(qx^2+qz^2)  s*(qy*qz-qw*qx)  0
                  s*(qx*qz-qw*qy)  s*(qx*qy+qw*qz)  1-s*(qy^2+qx^2)  0
                  0                0                0                1
                ]
                
            ref: https://blog.csdn.net/shuaiilong/article/details/22849225
        """
        q = self.Q
        self.Transform = np.array(
        [
            [1.0 - 2 * (q[1] * q[1] + q[2] * q[2]), 2 * (q[0] * q[1] - q[3] * q[2]), 2 * (q[3] * q[1] + q[0] * q[2]), 0],
            [2 * (q[0] * q[1] + q[3] * q[2]), 1.0 - 2 * (q[0] * q[0] + q[2] * q[2]), 2 * (q[1] * q[2] - q[3] * q[0]), 0],
            [2 * (q[0] * q[2] - q[3] * q[1]), 2 * (q[1] * q[2] + q[3] * q[0]), 1.0 - 2 * (q[0] * q[0] + q[1] * q[1]), 0],
            [0,0,0,1]
        ])
        # q0, q1, q2, q3 = self.Q[0], self.Q[1], self.Q[2], self.Q[3]
        # self.Transform = np.array( \
        # [
        #    [1-2*q2**2-2*q3**2, 2*q1*q2+2*q0*q3, 2*q1*q3-2*q0*q2, 0],
        #    [2*q1*q2-2*q0*q3, 1-2*q1**2-2*q3**2, 2*q2*q3+2*q0*q1, 0],
        #    [2*q1*q3+2*q0*q2, 2*q2*q3-2*q0*q1, 1-2*q1**2-2*q2**2, 0],
        #    [0,                  0,              0,               1],
        # ])
        # Translation = np.eye(4)
        # Translation[:3, 3] = self.Translation
        # self.Transform = np.dot(Translation, self.Transform)
        self.Transform[:3, 3] = self.Translation
        return self.Transform
    
    def get_Transform_to_baselink(self):
        # the matrix transform camera coordinate to baselink
        self.Transform_to_baselink = np.dot(self.pT, self.Transform)
        return self.Transform_to_baselink
    
    def get_Transform_from_baselink(self):
        # the matrix transform baselink to camera coordinate
        self.Transform_from_baselink = np.linalg.inv(self.Transform_to_baselink)
        return self.Transform_from_baselink
    
class Ext_cam:
    """
        This class is to pack 4 extrinsic matrix of cameras together
        f_c: front camera
        f_r: front right camera
        f_l: front left camera
        b: back camera
    """
    def __init__(self):
        self.f_c = Extrinsic_Camera(np.array([0.0, 0.0, 0.0, -0.5070558775462676, 0.47615311808704197, -0.4812773544166568, 0.5334272708696808]))
        self.f_r = Extrinsic_Camera(np.array([-0.047226, -0.564235, -0.0429843, -0.13426, 0.75675, -0.63051, 0.10844]))
        self.f_l = Extrinsic_Camera(np.array([-0.073756,  0.56121, -0.029907, -0.78108, 0.098631, -0.077138, 0.61175]))
        self.b   = Extrinsic_Camera(np.array([-1.5138, -0.043561, -0.016857, 0.48989, 0.51567, -0.50218, -0.49185]))

ext_cam = Ext_cam()

if __name__ == "__main__":
    f_c = Extrinsic_Camera(np.array([0.0, 0.0, 0.0, -0.5070558775462676, 0.47615311808704197, -0.4812773544166568, 0.5334272708696808]))
    f_r = Extrinsic_Camera(np.array([0.559084, 0.0287952, -0.0950537, -0.0806252, 0.607127, 0.0356452, 0.789699]), f_c.Transform_to_baselink)
    f_l = Extrinsic_Camera(np.array([-0.564697, 0.0402756, -0.028059, -0.117199, -0.575476, -0.0686302, 0.806462]), f_c.Transform_to_baselink)
    b = Extrinsic_Camera(np.array([-1.2446, 0.21365, -0.91917, 0.074732, -0.794, -0.10595, 0.59393]), f_l.Transform_to_baselink)
    print(np.dot(f_l.Transform_to_baselink, np.array([0,0,0,1]).T))
    print(np.dot(b.Transform_to_baselink, np.array([0,0,0,1]).T))
    P = np.array([549.959, 0.0, 728.516, 0.0, 0.0, 549.851, 448.147, 0.0, 0.0, 0.0, 1.0, 0.0]).reshape(3,4)
    K = np.array([[658.929184246, 0.0, 721.005287695, 0.0, 658.798994733, 460.495402628, 0.0, 0.0, 1.0]]).reshape(3,3)
    _P = np.dot(K,f_l.Transform_to_baselink[:3,:3])
    print(P)
    print(np.linalg.inv(_P))
    # print(np.dot(K,f_l.Transform_to_baselink[:3,:3]))
    print(f_l.Transform_to_baselink)
    print(np.linalg.pinv(f_l.Transform_to_baselink))
    Tb_inv = np.linalg.pinv(f_l.Transform_to_baselink)

    """
    debugging:
        the origin of f_r at coordinate baselink is: 
            -0.047, -0.564, -0.043
        the origin of f_r at coordinate f is:
            0.559, 0.028, -0.095
        the origin of baselink at coordinate f_r is:
            -0.236, 0.0498, -0.514
    """