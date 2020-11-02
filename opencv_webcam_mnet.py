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

cap.set(3,640)
cap.set(4,480)

run_inference = False

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

        #image_np_with_detections = image.copy()

        viz_utils.visualize_boxes_and_labels_on_image_array(
            image,
            detections['detection_boxes'],
            detections['detection_classes'],
            detections['detection_scores'],
            category_index,
            use_normalized_coordinates=True,
            max_boxes_to_draw=200,
            min_score_thresh=.30,
            agnostic_mode=False)


    cv2.imshow('Inference',image)

    # Keyboard control
    key = cv2.waitKey(1) & 0xFF
    if key == ord(' '):
        print("starting inference")
        run_inference = True
    elif key == ord('p'):


        predictions = model.predict(img_array)
        print(predictions)
        score = tf.nn.softmax(predictions[0])
        print(
            "This image most likely belongs to {} with a {:.2f} percent confidence."
                .format(classes[np.argmax(score)], 100 * np.max(score))
        )

    elif key == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()