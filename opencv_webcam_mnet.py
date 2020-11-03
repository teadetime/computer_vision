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

warnings.filterwarnings('ignore')   # Suppress Matplotlib warnings
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils

IMAGE_DIR = "images/another"
IMAGE_PATHS = glob.glob(IMAGE_DIR+"/*")
print(IMAGE_PATHS)
PATH_TO_MODEL_DIR = "my_model_mnetv2"
PATH_TO_LABELS = "my_model_mnetv2/label_map.pbtxt"
PATH_TO_SAVED_MODEL = PATH_TO_MODEL_DIR + "/saved_model"

cap = cv2.VideoCapture(0)
print(cap.get(3), cap.get(4))

# # Set to lower res
# directory = pathlib.Path.cwd() / 'images'
# print(directory)
# models_dir = pathlib.Path.cwd() / 'models'

im_width = 640
im_height = 320
cap.set(3,im_width)
cap.set(4,im_height)

run_inference = False
prune = True
prune_num = 2
'''
SETUP THE MODEL
'''
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
while(True):
    # Capture frame-by-frame

    ret, frame = cap.read()

    # Our operations on the frame come here
    image = frame
    image_for_model = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # Display the resulting frame

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

        image_copy = image.copy()
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


        # Time to process the boxes
        for num, box in enumerate(selected_boxes):
            print(box)
            ymin = int(box[0] * im_height)
            xmin = int(box[1] * im_width)
            crop_h = int((box[2] - box[0]) * im_height)
            crop_w = int((box[3] - box[1]) * im_width)
            bbox = tf.image.crop_to_bounding_box(image,ymin,xmin, crop_h, crop_w).numpy()
            cv2.imshow('bbox_'+str(num), bbox)
    cv2.imshow('Inference',image)

    # Keyboard control
    key = cv2.waitKey(1) & 0xFF
    if key == ord(' '):
        print("starting inference")
        run_inference = not run_inference
    if key == ord('p'):
        print("Pruning bboxes to ", prune_num)
        prune = True#not prune
    if key == ord('o'):
        if prune_num == 1: prune_num = 2
        else: prune_num = 1
        print("Pruning bboxes to: ", prune_num)
    elif key == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()