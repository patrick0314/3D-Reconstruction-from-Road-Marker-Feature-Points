import numpy as np
import cv2
import matplotlib.pyplot as plt

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

# visualize 3d point cloud via plt
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


"""
debugging:
    the origin of f_r at coordinate baselink is: 
        -0.047, -0.564, -0.043
    the origin of f_r at coordinate f is:
        0.559, 0.028, -0.095
    the origin of baselink at coordinate f_r is:
        -0.236, 0.0498, -0.514
"""