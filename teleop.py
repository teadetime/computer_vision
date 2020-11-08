#!/usr/bin/env python3
from __future__ import print_function, division
import rospy
from neato_node.msg import Bump
from std_msgs.msg import Int8MultiArray
from geometry_msgs.msg import Twist, Vector3
from visualization_msgs.msg import Marker
import tty
import select
import sys
import termios
import cv2
import pathlib
import os
from tensorflow import keras
from PIL import Image
import time
import tensorflow as tf
import numpy as np
import glob

from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils

PATH_TO_MODEL_DIR = os.path.expanduser("~/catkin_ws/src/computer_vision/my_model_mnetv2")
PATH_TO_LABELS = os.path.expanduser("~/catkin_ws/src/computer_vision/my_model_mnetv2/label_map.pbtxt")
PATH_TO_SAVED_MODEL = PATH_TO_MODEL_DIR + "/saved_model"

''' SETUP THE MODEL'''
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


settings = termios.tcgetattr(sys.stdin)
key = None

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

    return frameFiltered

class tele(object):
    def __init__(self):
        rospy.init_node('tele_vision')
        # rospy.Subscriber('/bump', Int8MultiArray, self.process_bump)
        self.pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
        self.desired_vel = 0.0
        self.angular_vel = 0.0

    def getKey(self):
        tty.setraw(sys.stdin.fileno())
        select.select([sys.stdin], [], [], 0)
        key = sys.stdin.read(1)
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)
        return key

    # def process_bump(self, msg):
    #     print(msg)
    #     if any((msg.data[0], msg.data[1], msg.data[2], msg.data[3])):
    #         self.desired_velocity = 0.0
    #         print("stop")

    def run(self):
        r = rospy.Rate(10)
        print("Starting Program")
        run_inference = True
        prune = True
        prune_num = 1
        save_video = False
        use_saved = False
        # used to record the time when we processed last frame
        prev_frame_time = 0
        # used to record the time at which we processed current frame
        new_frame_time = 0
        # font which we will be using to display FPS
        font = cv2.FONT_HERSHEY_SIMPLEX
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
            im_width = 640  # TODO: These are hardcoded
            im_height = 480

        while not rospy.is_shutdown():
            '''
            OPEN CV READ FROM WEBCAM
            '''
            ret, frame = cap.read()

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

                if prune:
                    selected_indices = tf.image.non_max_suppression(
                        detections['detection_boxes'], detections['detection_scores'], prune_num, .60)
                    selected_boxes = tf.gather(detections['detection_boxes'], selected_indices).numpy()
                    selected_classes = tf.gather(detections['detection_classes'], selected_indices).numpy()
                    selected_scores = tf.gather(detections['detection_scores'], selected_indices).numpy()
                else:
                    selected_boxes = detections['detection_boxes']
                    selected_classes = detections['detection_classes']
                    selected_scores = detections['detection_scores']
                # Pull out the bboxes before annotating!
                # Time to process the boxes
                bboxes = []
                for num, box in enumerate(selected_boxes):
                    print(box)
                    # if selected_scores[num] < .7:
                    #     cv2.destroyWindow('bbox_'+str(num))
                    #     continue
                    ymin = int(box[0] * im_height)
                    xmin = int(box[1] * im_width)
                    crop_h = int((box[2] - box[0]) * im_height)
                    crop_w = int((box[3] - box[1]) * im_width)
                    bbox = tf.image.crop_to_bounding_box(image, ymin, xmin, crop_h, crop_w).numpy()
                    bboxes.append(bbox)
                    bboxFiltered = segmentHand(bbox)
                    contours, hierarchy = cv2.findContours(bboxFiltered, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE,
                                                           offset=(xmin, ymin))
                    origFrame = cv2.flip(frame, 1)
                    try:
                        contours = max(contours, key=lambda x: cv2.contourArea(x))
                        cv2.drawContours(origFrame, [contours], -1, (255, 255, 0), 2)
                        hull = cv2.convexHull(contours)
                        cv2.drawContours(origFrame, [hull], -1, (0, 255, 255), 2)
                    except ValueError:
                        pass
                    # print(np.shape(contours))

                    cv2.imshow("contours", cv2.flip(origFrame, 1))
                    cv2.imshow('bbox_' + str(num), bboxFiltered)
                    # cv2.imshow('bbox_'+str(num), bbox)

                # Visulaize the bboxes
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

            # Calculating the fps
            new_frame_time = time.time()
            fps = 1 / (new_frame_time - prev_frame_time)
            prev_frame_time = new_frame_time
            # converting the fps into integer
            fps = str(int(fps))
            # puting the FPS count on the frame
            cv2.putText(image, fps, (7, 70), font, 3, (100, 255, 0), 3, cv2.LINE_AA)
            cv2.imshow('Inference', image)

            # Keyboard control
            key = cv2.waitKey(1) & 0xFF
            if key == ord(' '):
                print("starting inference")
                run_inference = not run_inference
            if key == ord('p'):
                print("Pruning bboxes to ", prune_num)
                prune = True  # not prune
            if key == ord('o'):
                if prune_num == 1:
                    prune_num = 2
                else:
                    prune_num = 1
                print("Pruning bboxes to: ", prune_num)
            elif key == ord('q'):
                break




        #     self.pub.publish(Twist(linear=Vector3(x=self.desired_vel),angular=Vector3(z=self.angular_vel)))
        #     r.sleep()
        #     key = self.getKey()
        #     # USe arrow keys to increase velocity, space to stop
        #     if key == " ":
        #         self.desired_vel = 0
        #         self.angular_vel = 0
        #     elif key == 'A':
        #         self.desired_vel += .1
        #     elif key == "B":
        #         self.desired_vel -= .1
        #     if key == "C":
        #         self.angular_vel -= .1
        #     elif key == "D":
        #         self.angular_vel += .1
        self.pub.publish(Twist(linear=Vector3(x=0), angular=Vector3(z=0)))


if __name__ == '__main__':
    tele_nathan = tele()
    tele_nathan.run()

