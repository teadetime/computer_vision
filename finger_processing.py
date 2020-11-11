import cv2
import scipy.cluster.hierarchy as hcluster
import numpy as np


def segmentHand(frame):
    # Apply Color Thresholding to segment skin tones from backgrounds
    frameB = frame[:, :, 0]
    frameG = frame[:, :, 1]
    frameR = frame[:, :, 2]

    # Uniform Day light Thresholds
    ret, frameMaxMin = cv2.threshold(np.amax(frame, axis=2) - np.amin(frame, axis=2), 15, 255, cv2.THRESH_BINARY)  # >15
    ret, frameRminusG = cv2.threshold(abs(frameR - frameG), 15, 255, cv2.THRESH_BINARY)  # >15
    ret, frameRG = cv2.threshold(frameR - frameG, 0, 255, cv2.THRESH_BINARY)  # R>G
    ret, frameRB = cv2.threshold(frameR - frameB, 0, 255, cv2.THRESH_BINARY)  # R>B
    ret, frameGB = cv2.threshold(frameG - frameB, 0, 255, cv2.THRESH_BINARY)  # G>B
    ret, frameBfilt = cv2.threshold(frameB, 20, 255, cv2.THRESH_BINARY)  # >20
    ret, frameGfilt = cv2.threshold(frameG, 40, 255, cv2.THRESH_BINARY)  # >40
    ret, frameRfilt = cv2.threshold(frameR, 95, 255, cv2.THRESH_BINARY)  # >95
    floatSum = ((frameMaxMin.astype(float) + frameRminusG.astype(float) + frameRG.astype(float) + frameRB.astype(
        float) + frameGB.astype(float) + frameRfilt.astype(float) + frameGfilt.astype(float) + frameBfilt.astype(
        float)) / 255).astype(np.uint8)
    ret, frameFiltered = cv2.threshold(floatSum, 7, 8, cv2.THRESH_BINARY)
    frameFiltered[frameFiltered >= 8] = 255

    #Apply erosion and dilation techniques to lower noise within the mask
    frameFiltered = cv2.erode(frameFiltered, None, iterations=2)
    frameFiltered = cv2.dilate(frameFiltered, None, iterations=6)
    frameFiltered = cv2.erode(frameFiltered, None, iterations=2)

    return frameFiltered


def isFinger(pt1, pt2, pt3):
    '''
    Based on the hull defects start, far, and stop points,
    return whether a particular set of points matches the
    characteristics of a finger.
    '''
    threshold = 90  # degree threshold

    # Convert the tuple points to np arrays
    pt1 = np.asarray(pt1)
    pt2 = np.asarray(pt2)
    pt3 = np.asarray(pt3)

    # Take the difference between the center points and two other points
    pt21 = pt1 - pt2
    pt32 = pt3 - pt2

    # Calculate the angle between the three points
    cosine_angle = np.dot(pt21, pt32) / (np.linalg.norm(pt21) * np.linalg.norm(pt32))
    angle = np.arccos(cosine_angle)

    # If the angle between the three points is less than the threshold value, the center point is a finger tip
    if np.degrees(angle) < threshold:
        return True
    else:
        return False


def getFingertips(img, imgFilt, xOffset, yOffset):
    '''
    Given a binary mask, calculate the contour of the hand.
    Then, calculate the convex hull and corresponding convexity defects.
    Returns the total number of fingers and an annotated version of the
    image where finger tips are labelled.
    '''

    # Calculate contours and draw them on webcam frame
    contours, hierarchy = cv2.findContours(imgFilt, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE,
                                           offset=(xOffset, yOffset))  # RETR_TREE, RETR_EXTERNAL
    contours = max(contours, key=lambda x: cv2.contourArea(x))
    cv2.drawContours(img, [contours], -1, (255, 255, 0), 2)

    # Generate convex hull based on the contour
    hull = cv2.convexHull(contours, returnPoints=False)

    # Generate convexity defects based on convex hull
    defects = cv2.convexityDefects(contours, hull, False)

    # Create lists of starting, ending and far points of convexity defects
    points = []
    pointsX = []
    pointsY = []

    for i in range(defects.shape[0]):
        s, e, f, d = defects[i, 0]
        start = tuple(contours[s][0])
        end = tuple(contours[e][0])
        far = tuple(contours[f][0])

        # If given defect points match finger characteristics add viable points to the list
        if isFinger(start, far, end) and d > 5000:
            points.extend([start, end])
            pointsX.extend([start[0], end[0]])
            pointsY.extend([start[1], end[1]])

    # Use hierarchical clustering to match redundant points
    try:
        clusters = hcluster.fclusterdata(points, 20, criterion="distance")
    except:
        return 0, img
    numFingers = max(clusters)
    # Average X and Y values for points in the same cluster
    for i in range(numFingers):
        indices = np.asarray(np.where(clusters == i + 1)).astype(int)
        indices = indices[0]

        averagePointsX = np.take(pointsX, indices)
        averagePointsY = np.take(pointsY, indices)
        averagePointsX = round(np.average(averagePointsX))
        averagePointsY = round(np.average(averagePointsY))

        # Visualize Fingertip
        cv2.circle(img, (int(averagePointsX), int(averagePointsY)), 10, [0, 255, 255], -1)

    # Return number of fingers and annotated image
    return numFingers, img

