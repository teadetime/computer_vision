#!/usr/bin/env python3
import cv2
import pathlib
import os
from tensorflow import keras
from PIL import Image
import time
import tensorflow as tf
import numpy as np
import warnings
import glob
import math
warnings.filterwarnings('ignore')   # Suppress Matplotlib warnings
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils
import scipy.cluster.hierarchy as hcluster

IMAGE_DIR = "images/another"
IMAGE_PATHS = glob.glob(IMAGE_DIR+"/*")

PATH_TO_MODEL_DIR = "../computer_vision/my_model_mnetv2"
PATH_TO_LABELS = "../computer_vision/my_model_mnetv2/label_map.pbtxt"
PATH_TO_SAVED_MODEL = PATH_TO_MODEL_DIR + "/saved_model"
run_inference = False
prune = True
save_video = False
use_saved = False
visualizeBBox = False
visualizeContours = True

import sys
print(sys.path)
def segmentHand(frame):
    #Apply Color Thresholding to segment skin tones from backgrounds
    frameB = frame[:,:,0] 
    frameG = frame[:,:,1] 
    frameR = frame[:,:,2] 

    #Uniform Day light Thresholds
    ret, frameMaxMin = cv2.threshold(np.amax(frame, axis = 2)-np.amin(frame, axis = 2), 15,255, cv2.THRESH_BINARY) #>15
    ret, frameRminusG = cv2.threshold(abs(frameR-frameG), 15,255, cv2.THRESH_BINARY)#>15
    ret, frameRG = cv2.threshold(frameR - frameG, 0, 255, cv2.THRESH_BINARY) #R>G
    ret, frameRB = cv2.threshold(frameR - frameB, 0, 255, cv2.THRESH_BINARY) #R>B
    ret, frameGB = cv2.threshold(frameG - frameB, 0, 255, cv2.THRESH_BINARY) #G>B
    ret, frameBfilt = cv2.threshold(frameB, 20, 255, cv2.THRESH_BINARY) #>20
    ret, frameGfilt = cv2.threshold(frameG, 40, 255, cv2.THRESH_BINARY) #>40
    ret, frameRfilt = cv2.threshold(frameR, 95, 255, cv2.THRESH_BINARY) #>95
    floatSum = ((frameMaxMin.astype(float) + frameRminusG.astype(float) +frameRG.astype(float) + frameRB.astype(float) + frameGB.astype(float) + frameRfilt.astype(float) + frameGfilt.astype(float) + frameBfilt.astype(float))/255).astype(np.uint8)
    ret, frameFiltered = cv2.threshold(floatSum, 7, 8, cv2.THRESH_BINARY)
    frameFiltered[frameFiltered>=8] = 255

    
    frameFiltered = cv2.erode(frameFiltered, None, iterations=2)
    frameFiltered = cv2.dilate(frameFiltered, None, iterations=6)
    frameFiltered = cv2.erode(frameFiltered, None, iterations=2)
    
    return frameFiltered

def isFinger(pt1,pt2,pt3):
    '''
    Based on the hull defects start, far, and stop points, 
    return whether a particular set of points matches the
    characteristics of a finger.
    '''
    threshold = 90 #degree threshold

    #Convert the tuple points to np arrays
    pt1 = np.asarray(pt1)
    pt2 = np.asarray(pt2)
    pt3 = np.asarray(pt3)

    #Take the difference between the center points and two other points
    pt21 = pt1 - pt2
    pt32 = pt3 - pt2

    #Calculate the angle between the three points
    cosine_angle = np.dot(pt21, pt32) / (np.linalg.norm(pt21) * np.linalg.norm(pt32))
    angle = np.arccos(cosine_angle)

    #If the angle between the three points is less than the threshold value, the center point is a finger tip
    if np.degrees(angle)<threshold:
        return True
    else:
        return False


def getFingertips(img,imgFilt, xOffset, yOffset):
    '''
    Given a binary mask, calculate the contour of the hand. 
    Then, calculate the convex hull and corresponding convexity defects. 
    Returns the total number of fingers and an annotated version of the
    image where finger tips are labelled.
    '''

    #Calculate contours and draw them on webcam frame
    contours, hierarchy = cv2.findContours(imgFilt, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE, offset=(xOffset,yOffset)) #RETR_TREE, RETR_EXTERNAL
    contours = max(contours, key=lambda x: cv2.contourArea(x))
    cv2.drawContours(img, [contours], -1, (255,255,0), 2)

    #Generate convex hull based on the contour
    hull = cv2.convexHull(contours,returnPoints=False)
    
    #Generate convexity defects based on convex hull
    defects = cv2.convexityDefects(contours,hull,False)
    
    #Create lists of starting, ending and far points of convexity defects
    points = []
    pointsX = []
    pointsY = []
    
    for i in range(defects.shape[0]):
        s,e,f,d = defects[i,0]
        start = tuple(contours[s][0])
        end = tuple(contours[e][0])
        far = tuple(contours[f][0])

        #If given defect points match finger characteristics add viable points to the list 
        if isFinger(start,far,end) and d > 5000:
            points.extend([start,end])
            pointsX.extend([start[0],end[0]])
            pointsY.extend([start[1],end[1]])
    
    #Use hierarchical clustering to match redundant points
    try:
        clusters = hcluster.fclusterdata(points, 20, criterion="distance")
    except:
        return 0, img
    numFingers = max(clusters)
    #Average X and Y values for points in the same cluster
    for i in range(numFingers):
        indices = np.asarray(np.where(clusters== i+1)).astype(int)
        indices = indices[0]
        
        averagePointsX = np.take(pointsX,indices)
        averagePointsY = np.take(pointsY,indices)
        averagePointsX = round(np.average(averagePointsX))
        averagePointsY = round(np.average(averagePointsY))
       
        #Visualize Fingertip
        cv2.circle(img,(int(averagePointsX),int(averagePointsY)),10,[0,255,255],-1)
        
    #Return number of fingers and annotated image
    return numFingers, img
       

if not use_saved:
    cap = cv2.VideoCapture(0)
    print("Default image size")
    print(cap.get(3), cap.get(4))
    # # Set to lower res
    im_width = 640
    im_height = 480
    cap.set(3, im_width)
    cap.set(4, im_height)
else:
    # try playing a saved video
    cap = cv2.VideoCapture('output.avi')
    im_width = 640 #TODO: These are hardcoded
    im_height = 480

if save_video:
    # Define the codec and create VideoWriter object
    out = cv2.VideoWriter('output.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 20.0, (im_width,im_height))
prune_num = 1
'''
SETUP THE MODEL
'''
past_frames = []
avg_bbox_scores = []
num_past_frames = 30
print('Loading model...', end='')
start_time = time.time()
# Load saved model and build the detection function
detect_fn = tf.saved_model.load(PATH_TO_SAVED_MODEL)
end_time = time.time()
elapsed_time = end_time - start_time
print('Done! Took {} seconds'.format(elapsed_time))
# Load the labels
category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS,
                                         use_display_name=True)
'''
Image capture loop!
'''
# used to record the time when we processed last frame
prev_frame_time = 0

# used to record the time at which we processed current frame
new_frame_time = 0
# font which we will be using to display FPS
font = cv2.FONT_HERSHEY_SIMPLEX
while(True):
    # Capture frame-by-frame

    ret, frame = cap.read()
    if save_video:
        out.write(frame)
    # Our operations on the frame come here
    image = cv2.flip(frame, 1)
    image_for_model = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    if run_inference:
        # The input needs to be a tensor, convert it using `tf.convert_to_tensor`.
        input_tensor = tf.convert_to_tensor(image_for_model)
        # The model expects a batch of images, so add an axis with `tf.newaxis`.
        input_tensor = input_tensor[tf.newaxis, ...]

        # input_tensor = np.expand_dims(image_np, 0)
        detections = detect_fn(input_tensor)

        # All outputs are batches tensors.
        # Convert to numpy arrays, and take index [0] to remove the batch dimension.
        # We're only interested in the first num_detections.
        num_detections = int(detections.pop('num_detections'))
        detections = {key: value[0, :num_detections].numpy()
                      for key, value in detections.items()}
        detections['num_detections'] = num_detections

        # detection_classes should be ints.
        detections['detection_classes'] = detections['detection_classes'].astype(np.int64)


        selected_indices = tf.image.non_max_suppression(
            detections['detection_boxes'], detections['detection_scores'], 1, .60) # FORCE PRUNE TO 1 with >60% confidence
        selected_boxes = tf.gather(detections['detection_boxes'], selected_indices).numpy()
        selected_classes = tf.gather(detections['detection_classes'], selected_indices).numpy()
        selected_scores = tf.gather(detections['detection_scores'], selected_indices).numpy()
        # Pull out the bboxes before annotating!
        # Time to process the boxes
        bboxes = []
        # ONLY MESS WITH ONE BOX

        bbox_score = 0  # Add score to see if the bbox matches well with past bboxes)
        hand = selected_boxes[0]
        size_increase = .1 # Increase bbox 10% of the width and height

        ymin = int(hand[0] * im_height)
        xmin = int(hand[1] * im_width)
        crop_h = int((hand[2] - hand[0]) * im_height)
        crop_w = int((hand[3] - hand[1]) * im_width)
        center_x = (xmin + crop_w / 2) / im_width
        center_y = (ymin + crop_h / 2) / im_height

        # FIND THE ADJUSTMENT AMOUNT!
        add_w = size_increase * crop_w
        add_h = size_increase * crop_h

        # Apply new adjustments while keeping the box centered
        ymin -= add_h
        xmin -= add_w
        crop_w_inc = crop_w + 2 * add_w
        crop_h_inc = crop_h + 2 * add_h

        # Trim the values to the image!
        ymin = int(max(0,ymin))
        xmin = int(max(0, xmin))
        max_w = im_width-xmin
        max_h = im_height-ymin
        crop_w_inc = int(min(max_w, crop_w_inc))
        crop_h_inc = int(min(max_h, crop_h_inc))

        new_selected_boxes = np.copy(selected_boxes)
        new_selected_boxes[0][0] = ymin/im_height
        new_selected_boxes[0][1] = xmin/im_width
        new_selected_boxes[0][2] = (ymin+crop_h_inc) /im_height
        new_selected_boxes[0][3] = (xmin+crop_w_inc) /im_width
        # Write info about box into the past frames
        info_dict = {"bbox":hand, "cent_x":center_x, "cent_y":center_y, "score":selected_scores[0]}

        # Generate scores to determine if this is a jumpy box or a good one
        area = (crop_h/im_height) * (crop_w/im_width) / .45  # arbitrary scaling so that we are looking for big hands!
        area_score = area  # COuld apply another function here
        ml_score = 2 * float(info_dict["score"]) ** 3

        # Look through past frames and see how good of a match you have
        if not past_frames:
            # First run or missing frames
            bbox_score = 0
        else:
            for frme in past_frames:
                # Check to see if the center or xy is close
                x_off = abs(frme["cent_x"] - info_dict["cent_x"])
                y_off = abs(frme["cent_y"] - info_dict["cent_y"])
                x_score = 1/math.e**(x_off**(1/3))
                y_score = 1/math.e**(y_off**(1/3))
                frame_score = x_score+y_score+area_score+ml_score # Max should be around 5
                bbox_score += frame_score
        #Print score output so that we can see how it fluctuates
        cv2.putText(image, "BBox Score: " + str(int(bbox_score)), (7, 170), font, 1, (120, 205, 40), 3, cv2.LINE_AA)
        # Add frame to past frames
        past_frames.append(info_dict)
        avg_bbox_scores.append(bbox_score)
        # Look at the average bbox score of the past frames
        if not avg_bbox_scores:
            avg_bbox = 0
        else:
            avg_bbox = sum(avg_bbox_scores)/len(avg_bbox_scores)
        # If the bbox is very different from the average then it is a jumpy frame
        good_box = True
        if bbox_score < avg_bbox*.9 or info_dict["score"] < .5:
            # Not a great match
            print('THis frame isnt stable yet!!')
            good_box = False
        else:
            print("GOOD FRAME")

        # Only Process bboxes etc id the box is good
        if good_box:
            # CROP THE BOUNDING BOX
            # print(ymin, xmin, crop_h, crop_w)
            bbox = tf.image.crop_to_bounding_box(image,ymin,xmin, crop_h_inc, crop_w_inc).numpy()
            bboxes.append(bbox)

            #Segment the hand with a binary mask
            bboxFiltered = segmentHand(bbox)

            #Returns 0 if no fingers detected (also for 1 finger) or number of other fingers
            numFingers, image = getFingertips(image, bboxFiltered, xmin, ymin)

            numFingers = min(numFingers,5)
            cv2.putText(image, "Num Fingers: " +str(numFingers), (7, 400), font, 1, (100, 255, 0), 3, cv2.LINE_AA)
            
            

            # Visualize the bboxes
            if visualizeBBox:
                viz_utils.visualize_boxes_and_labels_on_image_array(
                    image,
                    selected_boxes,
                    selected_classes,
                    selected_scores,
                    category_index,
                    use_normalized_coordinates=True,
                    max_boxes_to_draw=200,
                    min_score_thresh=.50,
                    agnostic_mode=False)
                viz_utils.visualize_boxes_and_labels_on_image_array(
                    image,
                    new_selected_boxes,
                    selected_classes,
                    selected_scores,
                    category_index,
                    use_normalized_coordinates=True,
                    max_boxes_to_draw=200,
                    min_score_thresh=.50,
                    agnostic_mode=False)

                cv2.imshow('bbox_'+str(1), bboxFiltered)
            
        # Remove the first entry of the past frames and scores as long as they aren't empty
        if len(past_frames) > num_past_frames:
            del past_frames[0]
            del avg_bbox_scores[0]
    # Calculating the fps
    new_frame_time = time.time()
    fps = 1 / (new_frame_time - prev_frame_time)
    prev_frame_time = new_frame_time
    # converting the fps into integer
    fps = str(int(fps))
    # puting the FPS count on the frame
    cv2.putText(image, "FPS: " + fps, (7, 70), font, 1, (100, 255, 0), 3, cv2.LINE_AA)
    cv2.imshow('Inference', image)


    # Keyboard control
    key = cv2.waitKey(1) & 0xFF
    if key == ord(' '):
        print("starting inference")
        run_inference = not run_inference
    # if key == ord('p'):
    #     print("Pruning bboxes to ", prune_num)
    #     prune = True#not prune
    # if key == ord('o'):
    #     if prune_num == 1: prune_num = 2
    #     else: prune_num = 1
    #     print("Pruning bboxes to: ", prune_num)
    elif key == ord('q'):
        break

# When everything done, release the capture
cap.release()
# out.release()
cv2.destroyAllWindows()