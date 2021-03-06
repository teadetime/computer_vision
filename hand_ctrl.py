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
import math

import os
import time
import tensorflow as tf
import numpy as np
import finger_processing

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

class tele(object):
    def __init__(self):
        rospy.init_node('hand_ctrl')
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
        position_control = False
        visualizeBBox = False
        visualizeContours = True
        prune = True
        prune_num = 1
        save_video = False
        use_saved = False
        past_frames = []
        past_bbox_scores = []
        frames_in_gesture = 10
        finger_num_thresh = 7 # 3 of the 5 stored frames need to be the same!!
        past_fingers = []
        lin_v = 0
        ang_v = 0

        last_good_box = None
        numFingers = None

        num_past_frames = 30
        # used to record the time when we processed last frame
        prev_frame_time = 0
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
            '''OPEN CV READ FROM WEBCAM'''
            ret, frame = cap.read()

            # Our operations on the frame come here
            image = cv2.flip(frame, 1)
            image_for_model = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            if run_inference:
                # Use helper function to run the image through the model
                detections = self.gen_detections(image_for_model)
                # Process detections down to just one bbox
                selected_boxes, selected_classes, selected_scores = self.prune_detections(detections)


                bbox_score = 0  # Score to "Validate if this is a good inference
                hand = selected_boxes[0]
                size_increase = .1  # Increase bbox 10% of the width and height
                box_h, box_w, center_x, center_y, crop_h, crop_w, xmin, ymin = self.process_box(hand, im_height, im_width, size_increase)

                # Write info about box into the past frames
                info_dict = {"bbox": hand, "cent_x": center_x, "cent_y": center_y, "score": selected_scores[0]}

                # Generate scores to determine if this is a jumpy box or a good one
                area = (box_h) * (box_w) / .45  # arbitrary scaling so that we are looking for big hands!
                area_score = area  # Could apply another function here
                ml_score = 2 * float(info_dict["score"]) ** 3

                """
                SCORE THE IMAGE!!
                """
                # Look through past frames and see how good of a match you have
                if not past_frames:# First run or missing frames
                    bbox_score = 0
                else:
                    for frme in past_frames:
                        frame_score = self.compute_score(area_score, frme, info_dict, ml_score)
                        bbox_score += frame_score
                # Print score output so that we can see how it fluctuates
<<<<<<< HEAD
                cv2.putText(image, "BBox score: "+str(int(bbox_score)), (7, 435), font, 1, (100, 255, 0), 3, cv2.LINE_AA)
=======
                cv2.putText(image, "BBox Score: " + str(int(bbox_score)), (7, 435), font, 1, (100, 255, 0), 3, cv2.LINE_AA)
>>>>>>> e015b886190a2158fdf23cb0538ac7e4063bb21c

                # Add frame to past frames
                past_frames.append(info_dict)
                # Look at the average bbox score of the past frames
                if not past_bbox_scores:
                    avg_bbox = 0
                else:
                    avg_bbox = sum(past_bbox_scores) / len(past_bbox_scores)
                # If the bbox is very different from the average then it is a jumpy frame
                past_bbox_scores.append(bbox_score)
                good_box = True
                if bbox_score < avg_bbox * .9 or info_dict["score"] < .5:
                    # Not a great match
                    print('THis frame isnt stable yet!!')
                    good_box = False
                else:
                    print("GOOD FRAME")

                # Only Process bboxes etc id the box is good
                if good_box:
                    # Save the last good box so that it can be used
                    # CROP THE BOUNDING BOX
                    bbox = tf.image.crop_to_bounding_box(image, ymin, xmin, crop_h, crop_w).numpy()
                    # Segment the hand with a binary mask
                    bboxFiltered = finger_processing.segmentHand(bbox)

                    # Returns 0 if no fingers detected (also for 1 finger) or number of other fingers
                    numFingers, image = finger_processing.getFingertips(image, bboxFiltered, xmin, ymin)

                    numFingers = min(numFingers, 5)
                    cv2.putText(image, "Num Fingers: " + str(numFingers), (7, 400), font, 1, (100, 255, 0), 3,
                                cv2.LINE_AA)

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
                        cv2.imshow('bbox_' + str(1), bboxFiltered)
                    # num_fingers = int((last_good_box["cent_x"]-.5) * 4)
                    # print(num_fingers)
                else:
                    #num_fingers = None
                    # Use the last good box
                    pass

                # Switch to localization control
                # Process the xy position as long as a frame has been read
                if last_good_box:
                    # Subtract .5 so that the range is from 0-1 this is used to control the neato
                    x_pos = -last_good_box["cent_x"]+.5
                    y_pos = -last_good_box["cent_y"]+.5
                else:
                    x_pos = 0
                    y_pos = 0

                # Remove the first entry of the past frames and scores as long as they aren't empty
                if len(past_frames) > num_past_frames:
                    del past_frames[0]
                    del past_bbox_scores[0]
            # Calculating the fps
            new_frame_time = time.time()
            fps = 1 / (new_frame_time - prev_frame_time)
            prev_frame_time = new_frame_time
            # converting the fps into integer
            fps = str(int(fps))
            # puting the FPS count on the frame
<<<<<<< HEAD
            cv2.putText(image, "FPS: "+fps, (7, 470), font, 1, (100, 255, 0), 3, cv2.LINE_AA)
=======
            cv2.putText(image, "FPS: " + fps, (7, 470), font, 1, (100, 255, 0), 3, cv2.LINE_AA)
>>>>>>> e015b886190a2158fdf23cb0538ac7e4063bb21c
            cv2.imshow('Inference', image)

            # Keyboard control
            key = cv2.waitKey(1) & 0xFF
            if key == ord(' '):
                print("starting inference")
                run_inference = not run_inference
            if key == ord('x'):
                print("Switching control Style")
                position_control = not position_control
            elif key == ord('q'):
                break


            ## Simple XY COntrol
            # X translation controls angular velocity, Y position controls linear
            num_match = 0
            print(numFingers)
            for past_finger in past_fingers:
                if past_finger == numFingers:
                    num_match += 1
            past_fingers.append(numFingers)
            if len(past_fingers) > frames_in_gesture:
                past_fingers[0]
            if num_match < finger_num_thresh:
                numFingers = None # Then don't do anyhting for this frame
            if numFingers is None:
                # This means that the last frame wasn't good, maybe we should leave it
                print("No new hand detected, continuing")
            elif numFingers == 0:
                lin_v = 0
                ang_v = 0
            elif numFingers == 4:
                lin_v = -1
                ang_v = 0
            elif numFingers == 5:
                lin_v = 1
                ang_v = 0
            elif numFingers == 3:
                lin_v = 0
                ang_v = 1
            elif numFingers == 2:
                lin_v = 0
                ang_v = 1
            if position_control:
                lin_v = y_pos
                ang_v = x_pos

            self.pub.publish(Twist(linear=Vector3(x=lin_v), angular=Vector3(z=ang_v)))

    def compute_score(self, area_score, past_frame, curr_frame, box_score):
        # Check to see if the center or xy is close
        x_off = abs(past_frame["cent_x"] - curr_frame["cent_x"])
        y_off = abs(past_frame["cent_y"] - curr_frame["cent_y"])
        x_score = 1 / math.e ** (x_off ** (1 / 3))
        y_score = 1 / math.e ** (y_off ** (1 / 3))
        frame_score = x_score + y_score + area_score + box_score  # Max should be around 5
        return frame_score

    def process_box(self, hand, im_height, im_width, size_increase):
        box_h = (hand[2] - hand[0])
        box_w = (hand[3] - hand[1])
        # Center in range [0,1]
        center_x = (hand[1] + box_w / 2)
        center_y = (hand[0] + box_h / 2)
        add_w = size_increase * box_w
        add_h = size_increase * box_h
        ymin = int((hand[0] - add_h) * im_height)
        xmin = int((hand[1] - add_w) * im_width)
        ymax = int((hand[2] + add_h) * im_height)
        xmax = int((hand[3] + add_w) * im_width)
        ymin = max(0, ymin)
        xmin = max(0, xmin)
        ymax = min(ymax, im_height)
        xmax = min(xmax, im_width)
        crop_w = int(xmax - xmin)
        crop_h = int(ymax - ymin)
        hand[0] = ymin / im_height
        hand[1] = xmin / im_width
        hand[2] = ymax / im_height
        hand[3] = xmax / im_width
        return box_h, box_w, center_x, center_y, crop_h, crop_w, xmin, ymin

    def prune_detections(self, detections, prune_num=1):
        selected_indices = tf.image.non_max_suppression(
            detections['detection_boxes'], detections['detection_scores'], prune_num,
            .60)  # FORCE PRUNE TO 1 with >60% confidence
        selected_boxes = tf.gather(detections['detection_boxes'], selected_indices).numpy()
        selected_classes = tf.gather(detections['detection_classes'], selected_indices).numpy()
        selected_scores = tf.gather(detections['detection_scores'], selected_indices).numpy()
        return selected_boxes, selected_classes, selected_scores

    def gen_detections(self, in_image):
        input_tensor = tf.convert_to_tensor(in_image)
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
        return detections


if __name__ == '__main__':
    tele_nathan = tele()
    tele_nathan.run()

