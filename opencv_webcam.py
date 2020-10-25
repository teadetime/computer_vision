import numpy as np
import cv2
import pathlib
import glob
import os

cap = cv2.VideoCapture(0)
print(cap.get(3), cap.get(4))

# Set to lower res
directory = pathlib.Path('/home/nathan/catkin_ws/src/computer_vision/images/')
classes = ['hand', 'something', 'another'].sort()
im_class = 0


cap.set(3,640)
cap.set(4,480)
def mk_dirs(dir, list, suffix=''):
    for folder in list:
        new_folder = dir/(folder+suffix)
        if not os.path.exists(new_folder):
            os.makedirs(new_folder)

def get_img_num(path, exten='.jpeg'):
    list_of_files = glob.glob(str(directory / classes[im_class]) + "/*")
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
    image = frame#cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
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
        print("switching class Right")
    elif key == ord('a'):
        cv2.destroyWindow(classes[im_class])
        im_class -= 1
        if im_class < 0: im_class = len(classes)-1  # Set to end class
        image_num = get_img_num(directory / classes[im_class])+1
        print("Swithing class Left")
    elif key == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()