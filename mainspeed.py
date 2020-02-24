#! /usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, print_function, absolute_import
import os
import datetime
from timeit import time
import warnings
import cv2
import numpy as np
import argparse
from PIL import Image
from yolo import YOLO
from deep_sort import preprocessing
from deep_sort import nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from tools import generate_detections as gdet
from deep_sort.detection import Detection as ddet
from collections import deque
from keras import backend

#ddding
backend.clear_session()
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input",help="path to input video", default = "./test_video/det_t1_video_00315_test.avi")
ap.add_argument("-c", "--class",help="name of class", default = "person")
args = vars(ap.parse_args())

pts = [deque(maxlen=10) for _ in range(9999)]
bts = [deque(maxlen=2) for _ in range(9999)]
frame_count = [0]*1000
speed = [-1]*9999
warnings.filterwarnings('ignore')
line = [(731, 302), (1006, 773)]
# initialize a list of colors to represent each possible class label
np.random.seed(100)
COLORS = np.random.randint(0, 255, size=(200, 3),
	dtype="uint8")
violation_id = [False] * 1000
mouse_frame = None
click = False     # Mouse 클릭된 상태 (false = 클릭 x / true = 클릭 o) : 마우스 눌렀을때 true로, 뗏을때 false로
mouse_x, mouse_y= -1,-1
highspeed= [0]
def ccw(A,B,C):
	return (C[1]-A[1]) * (B[0]-A[0]) > (B[1]-A[1]) * (C[0]-A[0])


def intersect(A,B,C,D):
	return ccw(A,C,D) != ccw(B,C,D) and ccw(A,B,C) != ccw(A,B,D)


def draw_line(event, x, y, flags, param):
    global mouse_x, mouse_y, click, line  # 전역변수 사용

    if event == cv2.EVENT_LBUTTONDOWN:  # 마우스를 누른 상태
        click = True
        mouse_x, mouse_y = x, y
        print("first dot: (" + str(mouse_x) + ", " + str(mouse_y) + ")")

    elif event == cv2.EVENT_LBUTTONUP:
        click = False;  # 마우스를 때면 상태 변경
        cv2.line(mouse_frame, (mouse_x, mouse_y), (x, y), (255, 0, 0), 2)
        line = [(mouse_x, mouse_y), (x, y)]
        print("violation line: (" + str(mouse_x) + ", " + str(mouse_y) + ")" + "(" + str(x) + ", " + str(y) + ")")

def draw_Aline(event, x, y, flags, param):
    global mouse_x, mouse_y, click, A_line  # 전역변수 사용

    if event == cv2.EVENT_LBUTTONDOWN:  # 마우스를 누른 상태
        click = True
        mouse_x, mouse_y = x, y
        print("first dot: (" + str(mouse_x) + ", " + str(mouse_y) + ")")

    elif event == cv2.EVENT_LBUTTONUP:
        click = False;  # 마우스를 때면 상태 변경
        cv2.line(mouse_frame, (mouse_x, mouse_y), (x, y), (255, 0, 0), 2)
        A_line = [(mouse_x, mouse_y), (x, y)]
        print("violation line: (" + str(mouse_x) + ", " + str(mouse_y) + ")" + "(" + str(x) + ", " + str(y) + ")")

def draw_Bline(event, x, y, flags, param):
    global mouse_x, mouse_y, click, B_line  # 전역변수 사용

    if event == cv2.EVENT_LBUTTONDOWN:  # 마우스를 누른 상태
        click = True
        mouse_x, mouse_y = x, y
        print("first dot: (" + str(mouse_x) + ", " + str(mouse_y) + ")")

    elif event == cv2.EVENT_LBUTTONUP:
        click = False;  # 마우스를 때면 상태 변경
        cv2.line(mouse_frame, (mouse_x, mouse_y), (x, y), (255, 0, 0), 2)
        B_line = [(mouse_x, mouse_y), (x, y)]
        print("violation line: (" + str(mouse_x) + ", " + str(mouse_y) + ")" + "(" + str(x) + ", " + str(y) + ")")


def main(yolo):

    start = time.time()
    #Definition of the parameters
    max_cosine_distance = 0.5#0.9 余弦距离的控制阈值
    nn_budget = None
    nms_max_overlap = 0.3 #非极大抑制的阈值
    vio_counter = 0
    counter = []
    #frame counting


    #deep_sort
    model_filename = 'model_data/market1501.pb'
    encoder = gdet.create_box_encoder(model_filename,batch_size=1)

    metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
    tracker = Tracker(metric)

    writeVideo_flag = True

    video_capture = cv2.VideoCapture(args["input"])

    video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    original_fps = video_capture.get(cv2.CAP_PROP_FPS)


    output_size=(200,200)
    fourcc = cv2.VideoWriter_fourcc('m','p','4','v')
    out2=cv2.VideoWriter('%s_output.mp4' % (args["input"].split('.')[0]), fourcc, original_fps, output_size)
    if writeVideo_flag:

        # Define the codec and create VideoWriter object
        w = int(video_capture.get(3))
        h = int(video_capture.get(4))
        #fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        out = cv2.VideoWriter('./output/'+args["input"][43:57]+ "_" + args["class"] + '_output.mp4', fourcc, original_fps, (w, h))
        list_file = open('detection.txt', 'w')
        frame_index = -1


    fps = 0.0


    first, mouse_frame = video_capture.read()
    cv2.namedWindow('DrawLine')
    cv2.resizeWindow('DrawLine', 1280, 720)



    while True:
        cv2.setMouseCallback('DrawLine', draw_line)

        cv2.imshow('DrawLine', mouse_frame)


        if cv2.waitKey(0) == ord('c'):
            break
    cv2.destroyAllWindows()
    ##
    first, mouse_frame = video_capture.read()
    cv2.namedWindow('DrawALine')
    cv2.resizeWindow('DrawALine', 1280, 720)


    while True:
        cv2.setMouseCallback('DrawALine', draw_Aline)

        cv2.imshow('DrawALine', mouse_frame)

        if cv2.waitKey(0) == ord('a'):
            break
    cv2.destroyAllWindows()
    ##
    first, mouse_frame = video_capture.read()
    cv2.namedWindow('DrawBLine')
    cv2.resizeWindow('DrawBLine', 1280, 720)


    while True:
        cv2.setMouseCallback('DrawBLine', draw_Bline)

        cv2.imshow('DrawBLine', mouse_frame)

        if cv2.waitKey(0) == ord('b'):
            break
    cv2.destroyAllWindows()







    while True:


        ret, frame = video_capture.read()
        if ret != True:
            break

        t1 = time.time()





        image = Image.fromarray(frame[...,::-1]) #bgr to rgb
        boxs,class_names = yolo.detect_image(image)
        features = encoder(frame,boxs)
        # score to 1.0 here).
        detections = [Detection(bbox, 1.0, feature) for bbox, feature in zip(boxs, features)]
        # Run non-maxima suppression.
        boxes = np.array([d.tlwh for d in detections])
        scores = np.array([d.confidence for d in detections])
        indices = preprocessing.non_max_suppression(boxes, nms_max_overlap, scores)
        detections = [detections[i] for i in indices]

        # Call the tracker
        tracker.predict()
        tracker.update(detections)

        i = int(0)
        indexIDs = []
        c = []


        for det in detections:
            bbox = det.to_tlbr()






        for track in tracker.tracks:
            
            


            if not track.is_confirmed() or track.time_since_update > 1:
                continue

            

            
            
            # 상자그리기
            indexIDs.append(int(track.track_id))
            counter.append(int(track.track_id))
            bbox = track.to_tlbr()

            cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0,255,0), 2)
            cv2.putText(frame, str(track.track_id), (int(bbox[0]), int(bbox[1] - 50)), 0, 5e-3 * 100, (0,255,0), 2)
            if len(class_names) > 0:
                class_name = class_names[0]
                cv2.putText(frame, str(class_names[0]), (int(bbox[0]), int(bbox[1] - 20)), 0, 5e-3 * 100, (0,255,0), 2)

            i += 1
            # bbox_center_point(x,y)
            center = (int(((bbox[0]) + (bbox[2])) / 2), int(((bbox[1]) + (bbox[3])) / 2))
            # track_id[center]
            pts[track.track_id].append(center)
            bts[track.track_id].append(center)
            thickness = 2
            # center point
            cv2.circle(frame, (center), 1, (0,255,0), 2)


            #intersect A line
            for j in range(1, len(pts[track.track_id])):
                    if pts[track.track_id][0] is None or pts[track.track_id][1] is None:
                       continue
                    thickness = int(np.sqrt(64 / float(j + 1)) * 2)
                    cv2.line(frame,(pts[track.track_id][j-1]), (pts[track.track_id][j]),(255,0,0),2)
                    if intersect(pts[track.track_id][j - 1], pts[track.track_id][j], line[0], line[1]):
                        violation_id[track.track_id] = True
                    #if intersect(bts[track.track_id][0], bts[track.track_id][1], A_line[0], A_line[1]):
                        #frame_count[track.track_id] = frame_index

                    #if intersect(bts[track.track_id][0], bts[track.track_id][1], B_line[0], B_line[1]):
                        #if frame_index == frame_count[track.track_id]:
                            #continue
                        #speed[track.track_id]=324./(frame_index-frame_count[track.track_id])
                        #print(str(speed[track.track_id])+"km/h id:"+str(track.track_id))
                        #if speed[track.track_id] >20:
                            #highspeed.append(speed[track.track_id])
            #this is for speed meter
            for j in range(1, len(bts[track.track_id])):
                    if bts[track.track_id][0] is None or bts[track.track_id][1] is None:
                       continue
                    if intersect(bts[track.track_id][0], bts[track.track_id][1], A_line[0], A_line[1]):
                        frame_count[track.track_id] = frame_index

                    if intersect(bts[track.track_id][0], bts[track.track_id][1], B_line[0], B_line[1]):
                        if frame_index == frame_count[track.track_id]:
                            continue
                        speed[track.track_id]=324./(frame_index-frame_count[track.track_id])
                        print(str(speed[track.track_id])+"km/h id:"+str(track.track_id))
                        if speed[track.track_id] >20:
                            highspeed.append(speed[track.track_id])


            if violation_id[track.track_id] == True:

                indexIDs.append(int(track.track_id))
                counter.append(int(track.track_id))
                bbox = track.to_tlbr()

                cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 0, 255), 2)
                cv2.line(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 0, 255), 2)
                cv2.line(frame, (int(bbox[0]), int(bbox[3])), (int(bbox[2]), int(bbox[1])), (0, 0, 255), 2)
                cv2.putText(frame, str(track.track_id)+"offender", (int(bbox[0]), int(bbox[1] - 50)), 0, 5e-3 * 100, (0, 0, 255), 2)
                if len(class_names) > 0:
                    class_name = class_names[0]
                    cv2.putText(frame, str(class_names[0]), (int(bbox[0]), int(bbox[1] - 20)), 0, 5e-3 * 100, (0,255,255), 2)

                i += 1
                # bbox_center_point(x,y)
                center = (int(((bbox[0]) + (bbox[2])) / 2), int(((bbox[1]) + (bbox[3])) / 2))
                # track_id[center]
                pts[track.track_id].append(center)
                thickness = 2
                result_top = int(center[1]-output_size[1]/2)
                result_bottom = int(center[1]+output_size[1]/2)
                result_left = int(center[0]-output_size[0]/2)
                result_right = int(center[0]+output_size[0]/2)
                if result_top > 0 and result_bottom>0 and result_left>0 and result_right>0:
                    result_img = frame[result_top:result_bottom, result_left:result_right].copy()
                    out2.write(result_img)
                    cv2.imshow('result_img', result_img)



                
                
                
                
                
                # center point
                #cv2.circle(frame, (center), 1, (20,20,20), 1)
                #cv2.circle(frame, (center), 1, (20, 20, 20), thickness)
                        
            

        count = len(set(counter))

        vio_counter = violation_id.count(True)
        cv2.line(frame, line[0], line[1], (0, 255, 255), 2)
        cv2.line(frame, A_line[0], A_line[1], (0, 255, 255), 2)
        cv2.line(frame, B_line[0], B_line[1], (0, 255, 255), 2)
        #cv2.line(frame, line[0], line[1], (0, 255, 255), 1)
        cv2.putText(frame, "Speed meter:" + str(round(highspeed[len(highspeed)-1],2))+"km/h id:"+str(track.track_id), (int(20), int(180)),0, 5e-3 * 120, (0, 0 ,255),2)
        cv2.putText(frame, "Violated Counter: " + str(vio_counter), (int(20), int(150)),0, 5e-3 * 120, (0, 0 ,255),2)
        cv2.putText(frame, "Total Object Counter: "+str(count),(int(20), int(120)),0, 5e-3 * 120, (0,255,0),2)
        cv2.putText(frame, "Current Object Counter: "+str(i),(int(20), int(80)),0, 5e-3 * 120, (0,255,0),2)
        cv2.putText(frame, "FPS: %f"%(fps),(int(20), int(40)),0, 5e-3 * 100, (0,255,0),2)
        #cv2.putText(frame, "Violated Counter: " + str(vio_counter), (int(20), int(150)),0, 5e-3 * 100, (0, 0 ,255),1)
        #cv2.putText(frame, "Total Object Counter: "+str(count),(int(20), int(120)),0, 5e-3 * 100, (0,255,0),1)
        #cv2.putText(frame, "Current Object Counter: "+str(i),(int(20), int(80)),0, 5e-3 * 100, (0,255,0),1)
        #cv2.putText(frame, "FPS: %f"%(fps),(int(20), int(40)),0, 5e-3 * 100, (0,255,0),1)
        cv2.namedWindow("YOLO3_Deep_SORT", 0);
        cv2.resizeWindow('YOLO3_Deep_SORT', 1280, 720);
        cv2.imshow('YOLO3_Deep_SORT', frame)

        if writeVideo_flag:
            #save a frame
            out.write(frame)
            frame_index = frame_index + 1
            list_file.write(str(frame_index)+' ')
            if len(boxs) != 0:
                for i in range(0,len(boxs)):
                    list_file.write(str(boxs[i][0]) + ' '+str(boxs[i][1]) + ' '+str(boxs[i][2]) + ' '+str(boxs[i][3]) + ' ')
            list_file.write('\n')
        fps  = ( fps + (1./(time.time()-t1)) ) / 2
	#fpss  = 1./(time.time()-t1)
        #print(set(counter))

        # Press Q to stop!
        if cv2.waitKey(1) & 0xFF == ord('q'):
            #video_capture.stop()
            break
    print(" ")
    print("[Finish]")
    end = time.time()

    if len(pts[track.track_id]) != None:
       print(args["input"][43:57]+": "+ str(count) + " " + str(class_name) +' Found')

    else:
       print("[No Found]")

    video_capture.release()

    if writeVideo_flag:
        #video_capture.stop()
        out.release()
        list_file.close()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main(YOLO())
