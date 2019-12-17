from ctypes import *
import math
import random
import os
import cv2
import numpy as np
import time
import darknet
import matplotlib.pyplot as plt

def bb_center_dist(bb1, bb2):
    x1 = bb1[0]
    y1 = bb1[1]
    x2 = bb2[0]
    y2 = bb2[1]
    
    a = x1 - x2
    b = y1 - y2
    
    dist = math.sqrt((a * a) + (b * b))
    
    return dist
    
def bb_intersection_over_union(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    iou = interArea / float(boxAArea + boxBArea - interArea)
    
    centerX = int((xA + xB)/2)
    centerY = int((yA + yB)/2)

    startX = int((xA + centerX) / 2)
    startY = int((yA + centerY) / 2)

    endX = int((xB + centerX) / 2)
    endY = int((yB + centerY) / 2)
    
    return iou, (centerX, centerY), (startX, startY), (endX, endY)

def convertBack(x, y, w, h):
    xmin = int(round(x - (w / 2)))
    xmax = int(round(x + (w / 2)))
    ymin = int(round(y - (h / 2)))
    ymax = int(round(y + (h / 2)))
    return xmin, ymin, xmax, ymax
'''
def cvDrawBoxes(detections, img):
    #     print('detections length : ', len(detections))
    pt1 = (0, 0)
    pt2 = (0, 0)
    area = 0
    
    xmin = 0
    ymin = 0
    xmax = 0
    ymax = 0
    
    for detection in detections:
        if detection[0].decode() != 'car': # and detection[0].decode() != 'bus' and detection[0].decode() != 'truck':
            continue

        x, y, w, h = detection[2][0],\
            detection[2][1],\
            detection[2][2],\
            detection[2][3]
        xmin, ymin, xmax, ymax = convertBack( float(x), float(y), float(w), float(h))
        pt1 = (xmin, ymin)
        pt2 = (xmax, ymax)
        area = (ymax-ymin) * (xmax-xmin)
        #print('area:', area)
        if area >= 13000:
            cv2.rectangle(img, pt1, pt2, (0, 255, 0), 3)
        #         cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0, 255, 0), -1)
            cv2.putText(img,
                        detection[0].decode() +
                        " [" + str(round(detection[1] * 100, 2)) + "]",
                        (pt1[0], pt1[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        [0, 255, 0], 2)
        #area = (ymax-ymin) * (xmax-xmin)
    return img, area, (xmin, ymin, xmax, ymax)
'''

outputWidth = 1920
outputHeight = 1080
netMain = None
metaMain = None
altNames = None

def YOLO():
    global metaMain, netMain, altNames
    configPath = "./cfg/yolov3-spp.cfg"
    weightPath = "./photi-vision-data/yolov3-spp.weights"
    metaPath = "./cfg/coco.data"
    if not os.path.exists(configPath):
        raise ValueError("Invalid config path `" +
                         os.path.abspath(configPath)+"`")
    if not os.path.exists(weightPath):
        raise ValueError("Invalid weight path `" +
                         os.path.abspath(weightPath)+"`")
    if not os.path.exists(metaPath):
        raise ValueError("Invalid data file path `" +
                         os.path.abspath(metaPath)+"`")
    if netMain is None:
        netMain = darknet.load_net_custom(configPath.encode(
            "ascii"), weightPath.encode("ascii"), 0, 1)  # batch size = 1
    if metaMain is None:
        metaMain = darknet.load_meta(metaPath.encode("ascii"))
    if altNames is None:
        try:
            with open(metaPath) as metaFH:
                metaContents = metaFH.read()
                import re
                match = re.search("names *= *(.*)$", metaContents,
                                  re.IGNORECASE | re.MULTILINE)
                if match:
                    result = match.group(1)
                else:
                    result = None
                try:
                    if os.path.exists(result):
                        with open(result) as namesFH:
                            namesList = namesFH.read().strip().split("\n")
                            altNames = [x.strip() for x in namesList]
                except TypeError:
                    pass
        except Exception:
            pass
    #cap = cv2.VideoCapture(0)
    cap = cv2.VideoCapture("./photi-vision-data/parking04_edit.mp4")
    cap.set(3, outputWidth)
    cap.set(4, outputHeight)
    out = cv2.VideoWriter(
        "./photi-vision-data/output-parking04_edit.mp4", cv2.VideoWriter_fourcc(*"MP4V"), 30.0,
        (outputWidth, outputHeight))
    print("Starting the YOLO loop...")

    #darknet.set_gpu(0)
    # Create an image we reuse for each detect
    darknet_image = darknet.make_image(outputWidth,
                                       outputHeight,3)

    point1 = (0, 0)
    point2 = (0, 0)
    A = (point1, point2)
    AREA = 1
    X0, Y0, X1, Y1 = [0,0,0,0]
    rectangleA = (0, 0, 0, 0)

    #temp_images = []
    temp_rectangles = []
    temp_centers = []

    free_space_frames = 0

    while True:
        temp_centers.clear()
        temp_rectangles.clear()
        prev_time = time.time()
        ret, frame_read = cap.read()

        if ret == False:
            break

        frame_rgb = cv2.cvtColor(frame_read, cv2.COLOR_BGR2RGB)
        frame_resized = cv2.resize(frame_rgb,
                                   (outputWidth,
                                    outputHeight),
                                   interpolation=cv2.INTER_LINEAR)

        darknet.copy_image_from_bytes(darknet_image,frame_resized.tobytes())

        detections = darknet.detect_image(netMain, metaMain, darknet_image, thresh=0.25)
        #image, area, rectangleB = cvDrawBoxes(detections, frame_resized)
        image = frame_resized

        pt1 = (0, 0)
        pt2 = (0, 0)
        area = 0
    
        xmin = 0
        ymin = 0
        xmax = 0
        ymax = 0
        
        for detection in detections:
            if detection[0].decode() != 'car': # and detection[0].decode() != 'bus' and detection[0].decode() != 'truck':
                continue

            x, y, w, h = detection[2][0],\
                detection[2][1],\
                detection[2][2],\
                detection[2][3]
            xmin, ymin, xmax, ymax = convertBack( float(x), float(y), float(w), float(h))
            area = (ymax-ymin) * (xmax-xmin)
            pt1 = (xmin, ymin)
            pt2 = (xmax, ymax)
            
            if area >= 12000:
                cv2.rectangle(image, pt1, pt2, (0, 255, 0), 3)
                cv2.putText(image,
                        detection[0].decode() +
                        " [" + str(round(detection[1] * 100, 2)) + "]",
                        (pt1[0], pt1[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        [0, 255, 0], 2)
                temp_rectangles.append((xmin, ymin, xmax, ymax))
                temp_centers.append( (xmin + (xmax-xmin)/2, ymin + (ymax-ymin)/2) )

        num_of_bb = len(temp_rectangles)
        print("number of bounding box in current scene:"+str(len(temp_rectangles)))

        free_space = False

        for i in range(0, len(temp_rectangles)):
            for j in range(0, len(temp_rectangles)):
                try:
                    iou, center, start, end = bb_intersection_over_union(temp_rectangles[i], temp_rectangles[j])
                    dist = bb_center_dist(temp_centers[i], temp_centers[j])
                    #area_i = bb_area(temp_rectangles[i])
                    #area_j = bb_area(temp_rectangles[j])
                    #print('dist:', dist)
                    #print('::::', iou, '::::', center, '::::::', temp_rectangles[i] , '::::::', temp_rectangles[j])
                except ZeroDivisionError:
                    print("ZeroDivision")
                if iou <= 0.05 and iou > 0 and dist <= 320:
                    free_space = True
                    print("Free space exits!")
                    cv2.rectangle(image, start, end, (255,0,0), -1)
                    #cv2.line(image, (temp_rectangles[i][0], temp_rectangles[i][3]), (temp_rectangles[j][0], temp_rectangles[j][3]), (255,0,0), 2)
                    #cv2.line(image, (temp_rectangles[i][2], temp_rectangles[i][3]), (temp_rectangles[j][2], temp_rectangles[j][3]), (255,0,0), 2)

        if free_space:
            free_space_frames += 1
        else:
            free_space_frames = 0

        if free_space_frames > 10:
             print("Empty space exits!")
             font = cv2.FONT_HERSHEY_DUPLEX
             cv2.putText(image, f"SPACE AVAILABLE!", (10, 150), font, 3.0, (0, 255, 0), 2, cv2.FILLED)


        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        out.write(image)
        cv2.waitKey(3)

    cap.release()
    out.release()

if __name__ == "__main__":
    YOLO()
