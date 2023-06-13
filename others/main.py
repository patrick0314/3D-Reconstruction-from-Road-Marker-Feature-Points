import cv2
import numpy as np
from visualization import OdomPlot
from VisualOdometry import VisualOdometry
import config
import matplotlib.pyplot as plt
from o3dUtils import visualize_pcd, display_points3d
import pickle
import open3d as o3d


OdomPlt = OdomPlot()


Front_VO = VisualOdometry(config.Front_Cam,
                    config.feature_finder,
                    config.descriptors_computer,
                    config.matcher,
                    config.findEssentialMat,
                    isDebug=config.VO_Debug)
Back_VO = VisualOdometry(config.Back_Cam,
                    config.feature_finder,
                    config.descriptors_computer,
                    config.matcher,
                    config.findEssentialMat,
                    isDebug=config.VO_Debug)

pcd = o3d.geometry.PointCloud()
viewer = o3d.visualization.Visualizer()
viewer.create_window(width=960, height=540)

opt = viewer.get_render_option()
opt.show_coordinate_frame = True
opt.background_color = np.asarray([0, 0, 0])

# cap = cv2.VideoCapture('output_video1.mp4')
cap = cv2.VideoCapture('test2_front.avi')
isFirstFrame = True

plt_pos = []
while cap.isOpened():
    # Read current frame
    ret, frame = cap.read()
    if not ret:
        break

    #Alert, 0 need to set to actual timestamp, timestamp not done yet!
    Front_VO.computeAndUpdateOdom(frame,0)
    
    #Skip first frame
    if isFirstFrame:
        isFirstFrame=False
        continue

    if config.VO_Debug:
        frame = Front_VO.debugFrame[0]

    #Optional: Just to make it smaller so that it fit on my monitor.
    frame = cv2.resize(frame, ( 640,  480))

    # Draw the current frame Speed on screen
    position,_ = Front_VO.Odom.getPoseByEuclidean()
    x, y, z = position
    cv2.putText(frame, "x: "+str(x), (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(frame, "y: "+str(y), (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(frame, "z: "+str(z), (30, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    plt_pos.append([x,-z])
    # OdomPlt.OdomPlot(x,-z,y)
    cv2.imshow('Odometry', frame)
    # display_points3d(Front_VO.map ,pcd, viewer)
    

    # stop every frame, only continue if button was press, q to quit
    # key = cv2.waitKey(0) & 0xFF
    # if key == ord('q'):
    #     break
    # else:
    #     continue
    
    # continuous playing, if q was press to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

plt_pos = np.array(plt_pos)
plt.plot(plt_pos[:,0], plt_pos[:,1])
plt.show()

color = Front_VO.o3dcolor()
print(color)
visualize_pcd(Front_VO.map, color)
map = Front_VO.map
with open("MapnColor.pckl","wb") as f:
    pickle.dump([map, color, plt_pos], f)

# save to txt
