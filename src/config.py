import cv2
import Utility
import numpy as np

#Constant
categories = {0:'zebracross', 1:'stopline', 2:'arrow', 3:'junctionbox', 4:'other'}
categories_color = {0:(255, 255, 0), 1:(0, 255, 0), 2:(0, 0, 255), 3:(0, 255, 255), 4:(255, 0, 255)}

# path to dataset
camera_path = ".\ITRI_dataset\camera_info\lucid_cameras_x00"
dataset_path = ".\ITRI_dataset"
output_path = ".\pcd"

"""
The preprocess was not wrap in a func is because I wish that we can see and change the setting here directly,
instead of going into function everytime config,
Algorithm: dilate -> bitwise_not
    dilate was done to expand the mask, to remove feature around the edge of the car
    bitwise_not was done, because the mask provided have true for car, and false for other region.
    We want to remove the car, so it should be false for car, true for other region.
"""
Front_Cam = Utility.Camera(
    init_Pose = np.eye(4),
    resolution = [1440,928],
    camera_Matrix = [[661.949026684, 0.0, 720.264314891], [0.0, 662.758817961, 464.188882538], [0.0, 0.0, 1.0]],
    distortion_coefficients = [-0.0309599861474, 0.0195100168293, -0.0454086703952, 0.0244895806953],
    rectification_matrix = [[1.000000, 0.000000, 0.000000], [0.000000, 1.000000, 0.000000], [0.000000, 0.000000, 1.000000]],
    projection_matrix = [[539.36, 0.0, 721.262, 0.0], [0.0, 540.02, 464.54, 0.0], [0.0, 0.0, 1.0, 0.0]],
    mask = cv2.bitwise_not(
            cv2.dilate(
                cv2.imread(f'{camera_path}\gige_100_f_hdr_mask.png', cv2.IMREAD_GRAYSCALE), 
                np.ones((5,5), dtype=np.uint8)
        )
    )
)
Left_Cam = Utility.Camera(
    init_Pose = np.eye(4),
    resolution = [1440, 928],
    camera_Matrix = [[658.929184246, 0.0, 721.005287695], [0.0, 658.798994733, 460.495402628], [0.0, 0.0, 1.0]],
    distortion_coefficients = [-0.0040468506737, -0.0433305077484, 0.0204357876847, -0.00127388969373],
    rectification_matrix = [[1.000000, 0.000000, 0.000000], [0.000000, 1.000000, 0.000000], [0.000000, 0.000000, 1.000000]],
    projection_matrix = [[549.959, 0.0, 728.516, 0.0], [0.0, 549.851, 448.147, 0.0], [0.0, 0.0, 1.0, 0.0]],
    mask = cv2.bitwise_not(
            cv2.dilate(
                cv2.imread(f'{camera_path}\gige_100_fl_hdr_mask.png', cv2.IMREAD_GRAYSCALE),
                np.ones((5, 5), dtype=np.uint8)
        )
    )
)

Right_Cam = Utility.Camera(
    init_Pose = np.eye(4),
    resolution = [1440, 928],
    camera_Matrix = [[660.195664468, 0.0, 724.021995966], [0.0, 660.202323944, 467.498636505], [0.0, 0.0, 1.0]],
    distortion_coefficients = [-0.0273485687967, 0.0202959209357, -0.0499610225624, 0.0261513487397],
    rectification_matrix = [[1.000000, 0.000000, 0.000000], [0.000000, 1.000000, 0.000000], [0.000000, 0.000000, 1.000000]],
    projection_matrix = [[542.867, 0.0, 739.613, 0.0], [0.0, 542.872, 474.175, 0.0], [0.0, 0.0, 1.0, 0.0]],
    mask = cv2.bitwise_not(
            cv2.dilate(
                cv2.imread(f'{camera_path}\gige_100_fr_hdr_mask.png', cv2.IMREAD_GRAYSCALE),
                np.ones((5, 5), dtype=np.uint8)
        )
    )
)

Back_Cam = Utility.Camera(
    init_Pose = np.eye(4),
    resolution = [1440, 928],
    camera_Matrix = [[658.897676983, 0.0, 719.335668486], [0.0, 659.869992391, 468.32106087], [0.0, 0.0, 1.0]],
    distortion_coefficients = [-0.00790509510948, -0.0356504181626, 0.00803540169827, 0.0059685787996],
    rectification_matrix = [[1.000000, 0.000000, 0.000000], [0.000000, 1.000000, 0.000000], [0.000000, 0.000000, 1.000000]],
    projection_matrix = [[547.741, 0.0, 715.998, 0.0], [0.0, 548.549, 478.83, 0.0], [0.0, 0.0, 1.0, 0.0]],
    mask = cv2.bitwise_not(
            cv2.dilate(
                cv2.imread(f'{camera_path}\gige_100_b_hdr_mask.png', cv2.IMREAD_GRAYSCALE),
                np.ones((5, 5), dtype=np.uint8)
        )
    )
)

"""
Feature detector:
FAST,KLT

Descriptor computer:
BRIEF,LBP,HOG,DAISY

Detect and compute
ORB,SIFT,FREAK,AKAZE,SURF,BRISK

example: 
    feature_finder = cv2.FastFeatureDetector_create().detect
    descriptors_computer = cv2.xfeatures2d.BriefDescriptorExtractor_create().compute
"""
#feature
detectAndCompute = cv2.ORB_create(nfeatures=3000, scaleFactor=1.5, nlevels=8)
# sift
# detectAndCompute = cv2.SIFT_create(nfeatures=1000, nOctaveLayers=3, contrastThreshold=0.04, edgeThreshold=10, sigma=1.6)
feature_finder = detectAndCompute.detect
descriptors_computer = detectAndCompute.compute
matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True).match
# sift
# matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True).match
findEssentialMat = lambda p1,p2,camera_Matrix: cv2.findEssentialMat(
                p1,p2,cameraMatrix=camera_Matrix, method=cv2.RANSAC, prob=0.9999, threshold=3
            )

#Flag
isDebug = True
VO_Debug = True

#True off all Debug, if isDebug is false here
if not isDebug:
    #all variable in here should set to False
    VO_Debug = False