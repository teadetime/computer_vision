#!/usr/bin/env python3
import cv2
import numpy as np
import tensorflow as tf
tf.__version__

cap = cv2.VideoCapture(0)

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    #frame = cv2.blur(frame,(5,5))
    frame = cv2.resize(frame, None, None, fx=1, fy=1)

    frameB = frame[:,:,0] 
    frameG = frame[:,:,1] 
    frameR = frame[:,:,2] 
    ret, frameMaxMin = cv2.threshold(np.amax(frame, axis = 2)-np.amin(frame, axis = 2), 15,255, cv2.THRESH_BINARY) #>15
    ret, frameRminusG = cv2.threshold(abs(frameR-frameG), 15,255, cv2.THRESH_BINARY)#>15
    ret, frameRG = cv2.threshold(frameR - frameG, 0, 255, cv2.THRESH_BINARY)
    ret, frameRB = cv2.threshold(frameR - frameB, 0, 255, cv2.THRESH_BINARY)
    ret, frameGB = cv2.threshold(frameG - frameB, 0, 255, cv2.THRESH_BINARY)
    ret, frameB = cv2.threshold(frameB, 20, 255, cv2.THRESH_BINARY) #>20
    ret, frameG = cv2.threshold(frameG, 40, 255, cv2.THRESH_BINARY) #>40
    ret, frameR = cv2.threshold(frameR, 95, 255, cv2.THRESH_BINARY) #>95
    floatSum = ((frameMaxMin.astype(float) + frameRminusG.astype(float) +frameR+frameRG.astype(float) + frameRB.astype(float) + frameGB.astype(float) + frameMaxMin.astype(float) + frameRminusG.astype(float))/255).astype(np.uint8)
    ret, frameFiltered = cv2.threshold(floatSum, 7, 8, cv2.THRESH_BINARY)
    frameFiltered[frameFiltered>=8] = 255

    cv2.imshow('Masked',frameFiltered)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()



