#!/usr/bin/env python3
import cv2
import numpy as np
import scipy.cluster.hierarchy as hcluster

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

def isFinger(pt1,pt2,pt3):
    threshold = 60 #degree threshold

    pt1 = np.asarray(pt1)
    pt2 = np.asarray(pt2)
    pt3 = np.asarray(pt3)
    pt21 = pt1 - pt2
    pt32 = pt3 - pt2

    cosine_angle = np.dot(pt21, pt32) / (np.linalg.norm(pt21) * np.linalg.norm(pt32))
    angle = np.arccos(cosine_angle)

    if np.degrees(angle)<threshold:
        return True
    else:
        return False



src = cv2.imread(r'/home/sander/Documents/computer_vision/hand.jpg')
scale_percent = 20

#calculate the 50 percent of original dimensions
width = int(src.shape[1] * scale_percent / 100)
height = int(src.shape[0] * scale_percent / 100)
img = cv2.resize(src,dsize = (width,height))
imgFilt =  segmentHand(img)
contours, hierarchy = cv2.findContours(imgFilt, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE )
contours = max(contours, key=lambda x: cv2.contourArea(x))
cv2.drawContours(img, [contours], -1, (255,255,0), 2)
hull = cv2.convexHull(contours,returnPoints=False)
defects = cv2.convexityDefects(contours,hull,False)
points = []
pointsX = []
pointsY = []
startPT = []
endPT = []
farPT = []
for i in range(defects.shape[0]):
    s,e,f,d = defects[i,0]
    start = tuple(contours[s][0])
    end = tuple(contours[e][0])
    far = tuple(contours[f][0])
    points.extend([start,end,far])
    pointsX.extend([start[0],end[0],far[0]])
    pointsY.extend([start[1],end[1],far[1]])
    if np.linalg.norm(np.array(start)-np.array(far)) > 40:
        startPT.append(start)
        farPT.append(far)
        endPT.append(end)
    #cv2.line(img,start,end,[0,255,0],2)
    #cv2.line(img,start,far,[255,0,0],2)
    #cv2.line(img,far,end,[255,0,0],2)
    #print(end)
    #cv2.circle(img,far,5,[0,0,255],-1)
    #cv2.circle(img,start,5,[255,0,255],-1)
    #cv2.circle(img,end,5,[255,0,255],-1)

clusters = hcluster.fclusterdata(points, 20, criterion="distance") #20 Threshold hardcoded
'''
maxCluster = max(clusters)
clustersOrdered = clusters+max(clusters)
counter = 0
for i in range(len(clusters)):
    index = clustersOrdered[i]
    if index>maxCluster:
        clustersOrdered[clustersOrdered == index] = counter
        print(clustersOrdered)
        counter +=1
clusters = clustersOrdered
'''
pointsAvg = []
for i in range(max(clusters)):
    indices = np.asarray(np.where(clusters== i+1)).astype(int)
    indices = indices[0]
    averagePointsX = np.take(pointsX,indices)
    averagePointsY = np.take(pointsY,indices)
    pointsAvg.append((round(np.average(averagePointsX)),round(np.average(averagePointsY))))

pointsAvg.append(pointsAvg[0])
startPT.append(startPT[0])
farPT.append(farPT[0])
endPT.append(endPT[0])

avg = []
for i in range(1,len(startPT)):
    startX = startPT[i][0]
    startY = startPT[i][1]
    endX = endPT[i-1][0]
    endY = endPT[i-1][1]
    
    if np.linalg.norm(np.array(start)-np.array(far)) < 20:
        avgPT = tuple((round((startX+endX)/2),round((startY+endY)/2)))
        avg.append(avgPT)
  
numFingers = 0
for i in range(len(avg)):
    if isFinger(farPT[i],avg[i],farPT[i+1]):
        cv2.circle(img,avg[i],10,[0,255,255],-1)
        cv2.line(img,avg[i],farPT[i],[255,0,0],2)
        cv2.line(img,avg[i],farPT[i+1],[255,0,0],2)
        numFingers+=1
    else:
        pass
  
print(numFingers)

    

cv2.imshow('hand',img)
cv2.waitKey(0)  
  
#closing all open windows  
cv2.destroyAllWindows() 