import math
import os
import sys

import cv2
import numpy as np


class RoadMarkerDetection():
    def __init__(self, coefficient=0.0001) -> None:
        self.coefficient = coefficient

        self.points = [] # all points

    def detection(self, img, camera_mask, road_marker, config):
        self.points = []
        camera_mask = (camera_mask.reshape(camera_mask.shape[0], camera_mask.shape[1], 1).repeat(3, axis=2)) // 255
        img = img * camera_mask
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Compute mask in YCbCr color space
        img_YCbCr = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
        lower = np.array([130, 118, 118])
        upper = np.array([255, 138, 138])
        mask1 = cv2.inRange(img_YCbCr, lower, upper)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask2 = cv2.morphologyEx(mask1, cv2.MORPH_CLOSE, kernel=kernel, iterations=13)
        '''
        mask_all = np.concatenate((mask1, mask2), axis=1)
        mask_all = cv2.resize(mask_all, (1280, 640))
        cv2.imshow('mask', mask_all)
        cv2.waitKey(0)
        '''

        imgDetected = img.copy()
        imgContour = img.copy()
        pre_points = 0
        for i in range(5):
            # implement Non-Maximum Suppression
            bboxes = self.nms(np.array(road_marker[road_marker['class_id']==i]), 0.15)

            # find the contour and mark the corner points
            for bbox in bboxes:
                x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
                class_id, probability = int(bbox[4]), float(bbox[5])

                if probability < 0.1: continue

                # Write bounding box with class and probability
                if x1 < 0: x1 = 0
                if x2 < 0: x2 = 0
                if y1 < 0: y1 = 0
                if y2 < 0: y2 = 0
                cv2.line(imgDetected, (x1, y1), (x1, y2), config.categories_color[class_id], 3)
                cv2.line(imgDetected, (x1, y2), (x2, y2), config.categories_color[class_id], 3)
                cv2.line(imgDetected, (x2, y2), (x2, y1), config.categories_color[class_id], 3)
                cv2.line(imgDetected, (x2, y1), (x1, y1), config.categories_color[class_id], 3)
                cv2.putText(imgDetected, '{} {:.3f}'.format(config.categories[class_id], probability), (x1-3, y1-3), cv2.FONT_HERSHEY_PLAIN, 1.5, config.categories_color[class_id], 1, cv2.LINE_AA)
                    
                contours = self.detectContour(mask2, x1, x2, y1, y2)

                if not contours: continue # if there is no contour found
                imgContour = cv2.drawContours(imgDetected, contours, -1, config.categories_color[class_id], 1, lineType=cv2.LINE_AA)
                
                length = 2*(x2-x1) + 2*(y2-y1)
                self.detectCorner(contours, length, img, config, class_id)
                
                # if there is no key point detected in mask2, compute with mask1
                if len(self.points) - pre_points < 2:
                    contours = self.detectContour(mask1, x1, x2, y1, y2)
                    if not contours: continue # if there is no contour found
                    imgContour = cv2.drawContours(imgDetected, contours, -1, config.categories_color[class_id], 1, lineType=cv2.LINE_AA)
                    self.detectCorner(contours, length, img, config, class_id)

                pre_points = len(self.points)
        
        kp = self.toKeyPoint(self.points)
        imgKeyPoints = cv2.drawKeypoints(img, kp, None, color=(0, 255, 0), flags=0)
        #cv2.imwrite('../output/imgKeyPoints.jpg', imgKeyPoints)

        return kp, imgKeyPoints, imgContour
            
    def nms(self, bboxes, iou_thresh):
        x1 = bboxes[:, 0]
        y1 = bboxes[:, 1]
        x2 = bboxes[:, 2]
        y2 = bboxes[:, 3]
        areas = (y2 - y1) * (x2 - x1)
        
        result = []
        index = areas.argsort()
        while index.size > 0:
            i = index[0]
            result.append(i)
            
            x11 = np.maximum(x1[i], x1[index[1:]])
            y11 = np.maximum(y1[i], y1[index[1:]])
            x22 = np.minimum(x2[i], x2[index[1:]])
            y22 = np.minimum(y2[i], y2[index[1:]])
            w = np.maximum(0, x22 - x11 + 1)
            h = np.maximum(0, y22 - y11 + 1)
            overlaps = w * h
            ious = overlaps / (areas[i] + areas[index[1:]] - overlaps)
            
            idx = np.where(ious <= iou_thresh)[0]
            index = index[idx + 1]

        bboxes = bboxes[result]
        return bboxes
    
    def detectContour(self, mask, x1, x2, y1, y2):
        # Generate bounding box mask
        imgBounded = mask.copy()
        bounding_box = np.zeros(imgBounded.shape, dtype=np.uint8)
        bounding_box[y1:y2+1, x1:x2+1] = 1
        imgBounded *= bounding_box

        # Road marker corner points
        contours, hierarchy = cv2.findContours(imgBounded, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        return contours
    
    def detectCorner(self, contours, length, img, config, class_id):
        for contour in contours:
            if cv2.arcLength(contour, True) < (length/18): continue
            epsilon = self.coefficient * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            self.points.extend(approx[:, 0, :])
            img = cv2.drawContours(img, approx, -1, config.categories_color[class_id], 5)

    def toKeyPoint(self, points):
        points = np.array(points)
        step_size = 5
        kp = []
        for point in points: kp.append(cv2.KeyPoint(int(point[0]), int(point[1]), step_size))
        return kp