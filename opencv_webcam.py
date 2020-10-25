import numpy as np
import cv2
import pathlib
import glob
import os
import tensorflow as tf
from tensorflow import keras
from PIL import Image
cap = cv2.VideoCapture(0)
print(cap.get(3), cap.get(4))

# Set to lower res
directory = pathlib.Path.cwd() / 'images'
print(directory)
models_dir = pathlib.Path.cwd() / 'models'

input_model = None
if not input_model:
    input_model = max(glob.glob(os.path.join(models_dir, '*/')), key=os.path.getmtime)
print("Loading model from: ", input_model)
model = keras.models.load_model(input_model)

classes = ['another','hand', 'something']
classes.sort()
im_class = 0

cap.set(3,640)
cap.set(4,480)
def mk_dirs(dir, list, suffix=''):
    for folder in list:
        new_folder = dir/(folder+suffix)
        if not os.path.exists(new_folder):
            os.makedirs(new_folder)

def get_img_num(path, exten='.jpeg'):
    list_of_files = glob.glob(str(path) + "/*"+exten)
    try:
        latest_file = max(list_of_files, key=os.path.getctime)
        return int(latest_file.split('.')[0].split('_')[-1])# parse out the number
    except:
        return -1 # No files so return a negative 1
    #return len(glob.glob1(path, '*'+exten)) # Older implementation


mk_dirs(directory, classes)
image_num = get_img_num(directory / classes[im_class])+1
print(image_num)
# Image capture loop!
while(True):
    # Capture frame-by-frame

    ret, frame = cap.read()

    # Our operations on the frame come here
    image = frame#cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # Display the resulting frame
    cv2.imshow(classes[im_class],image)

    # Keyboard control
    key = cv2.waitKey(1) & 0xFF
    if key == ord(' '):
        print("Saving image num: ", image_num)
        file_loc = (directory / classes[im_class] / (classes[im_class] +'_' + str(image_num))).with_suffix('.jpeg')
        cv2.imwrite(str(file_loc), image)
        image_num = get_img_num(directory / classes[im_class])
        image_num += 1
    elif key == ord('d'):
        cv2.destroyWindow(classes[im_class])
        im_class += 1
        if im_class >= len(classes): im_class = 0
        image_num = get_img_num(directory/classes[im_class])+1
        print("Switching class Right")
    elif key == ord('a'):
        cv2.destroyWindow(classes[im_class])
        im_class -= 1
        if im_class < 0: im_class = len(classes)-1  # Set to end class
        image_num = get_img_num(directory / classes[im_class])+1
        print("Swithing class Left")
    elif key == ord('p'):
        # DO prediction on frame
        sunflower_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/592px-Red_sunflower.jpg"
        sunflower_path = tf.keras.utils.get_file('Red_sunflower', origin=sunflower_url)

        img = keras.preprocessing.image.load_img(
            sunflower_path, target_size=(480, 640)
        )
        print(type(img))
        swap_rbg = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(swap_rbg)
        print(type(img_pil))
        #

        img_array = keras.preprocessing.image.img_to_array(img_pil)
        img_array = tf.expand_dims(img_array, 0)  # Create a batch

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