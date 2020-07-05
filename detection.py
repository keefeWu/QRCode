#coding:utf-8
import cv2
import numpy as np
import math
def show(img, name = 'img'):
    maxHeight = 540
    maxWidth = 960
    scaleX = maxWidth / img.shape[1]  
    scaleY = maxHeight / img.shape[0]
    scale = min(scaleX, scaleY)
    if scale < 1:
        img = cv2.resize(img,(0,0),fx=scale, fy=scale)
    cv2.imshow(name,img)
    cv2.waitKey(0)



def convert_img_to_binary(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    binary_img = cv2.adaptiveThreshold(
            gray,
            255,                    # Value to assign
            cv2.ADAPTIVE_THRESH_MEAN_C,# Mean threshold
            cv2.THRESH_BINARY,
            11,                     # Block size of small area
            2,                      # Const to substract
        )
    return binary_img

def getContours(img):
    binary_img = convert_img_to_binary(img)
    # thresholdImage = binary_img
    thresholdImage = cv2.Canny(binary_img, 100, 200) #Edges by canny edge detection
    
    _, contours, hierarchy = cv2.findContours(
            thresholdImage, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    return thresholdImage, contours, hierarchy

def checkRatioOfContours(index, contours, hierarchy):
    firstChildIndex = hierarchy[0][index][2]
    secondChildIndex = hierarchy[0][firstChildIndex][2]
    firstArea = cv2.contourArea(contours[index]) / (
        cv2.contourArea(contours[firstChildIndex]) + 1e-5)
    secondArea = cv2.contourArea(contours[firstChildIndex]) / (
        cv2.contourArea(contours[secondChildIndex]) + 1e-5)
    return ((firstArea / (secondArea+ 1e-5)) > 1 and \
            ((firstArea / (secondArea+ 1e-5)) < 10))

def isPossibleCorner(contourIndex, levelsNum, contours, hierarchy):
    # if no chirld, return -1
    chirldIdx = hierarchy[0][contourIndex][2] 
    level = 0
    while chirldIdx != -1:
        level += 1
        chirldIdx = hierarchy[0][chirldIdx][2] 
    if level >= levelsNum:
        return checkRatioOfContours(contourIndex, contours, hierarchy)
    return False

def getContourWithinLevel(levelsNum, contours, hierarchy):
    # find contours has 3 levels
    patterns = []
    patternsIndices = []
    for contourIndex in range(len(contours)):
        if isPossibleCorner(contourIndex, levelsNum, contours, hierarchy):
            patterns.append(contours[contourIndex])
            patternsIndices.append(contourIndex)
    return patterns, patternsIndices

def isParentInList(intrestingPatternIdList, index, hierarchy):
    parentIdx = hierarchy[0][index][3]
    while (parentIdx != -1) and (parentIdx not in intrestingPatternIdList):
        parentIdx = hierarchy[0][parentIdx][3]
    return parentIdx != -1

def getOrientation(contours, centerOfMassList):
        distance_AB = np.linalg.norm(centerOfMassList[0].flatten() - centerOfMassList[1].flatten(), axis = 0)
        distance_BC = np.linalg.norm(centerOfMassList[1].flatten() - centerOfMassList[2].flatten(), axis = 0)
        distance_AC = np.linalg.norm(centerOfMassList[0].flatten() - centerOfMassList[2].flatten(), axis = 0)

        largestLine = np.argmax(
            np.array([distance_AB, distance_BC, distance_AC]))
        bottomLeftIdx = 0
        topLeftIdx = 1
        topRightIdx = 2
        if largestLine == 0:
            bottomLeftIdx, topLeftIdx, topRightIdx = 0, 2, 1 
        if largestLine == 1:
            bottomLeftIdx, topLeftIdx, topRightIdx = 1, 0, 2 
        if largestLine == 2:
            bottomLeftIdx, topLeftIdx, topRightIdx = 0, 1, 2 

        # distance between point to line:
        # abs(Ax0 + By0 + C)/sqrt(A^2+B^2)
        slope = (centerOfMassList[bottomLeftIdx][1] - centerOfMassList[topRightIdx][1]) / (centerOfMassList[bottomLeftIdx][0] - centerOfMassList[topRightIdx][0] + 1e-5)
        # y = kx + b => AX + BY +C = 0 => B = 1, A = -k, C = -b
        coefficientA = -slope
        coefficientB = 1
        constant = slope * centerOfMassList[bottomLeftIdx][0] - centerOfMassList[bottomLeftIdx][1]
        distance = (coefficientA * centerOfMassList[topLeftIdx][0] + coefficientB * centerOfMassList[topLeftIdx][1] + constant) / (
            np.sqrt(coefficientA ** 2 + coefficientB ** 2))


        pointList = np.zeros(shape=(3,2))
        # 回    回   tl   bl
        if (slope >= 0) and (distance >= 0):
            # if slope and distance are positive A is bottom while B is right
            if (centerOfMassList[bottomLeftIdx][0] > centerOfMassList[topRightIdx][0]):
                pointList[1] = centerOfMassList[bottomLeftIdx]
                pointList[2] = centerOfMassList[topRightIdx]
            else:
                pointList[1] = centerOfMassList[topRightIdx]
                pointList[2] = centerOfMassList[bottomLeftIdx]
            # TopContour in the SouthWest of the picture
            ORIENTATION = "SouthWest"

        # 回   回     bl     tl
        #
        #      回            tr
        elif (slope > 0) and (distance < 0):
            # if slope is positive and distance is negative then B is bottom
            # while A is right
            if (centerOfMassList[bottomLeftIdx][1] > centerOfMassList[topRightIdx][1]):
                pointList[2] = centerOfMassList[bottomLeftIdx]
                pointList[1] = centerOfMassList[topRightIdx]
            else:
                pointList[2] = centerOfMassList[topRightIdx]
                pointList[1] = centerOfMassList[bottomLeftIdx]
            ORIENTATION = "NorthEast"


        #       回            bl
        #
        # 回    回      tr    tl
        elif (slope < 0) and (distance > 0):
            if (centerOfMassList[bottomLeftIdx][0] > centerOfMassList[topRightIdx][0]):
                pointList[1] = centerOfMassList[bottomLeftIdx]
                pointList[2] = centerOfMassList[topRightIdx]
            else:
                pointList[1] = centerOfMassList[topRightIdx]
                pointList[2] = centerOfMassList[bottomLeftIdx]
            ORIENTATION = "SouthEast"
        # 回    回    tl   tr
        #
        # 回          bl
        elif (slope < 0) and (distance < 0):

            if (centerOfMassList[bottomLeftIdx][0] > centerOfMassList[topRightIdx][0]):
                pointList[2] = centerOfMassList[bottomLeftIdx]
                pointList[1] = centerOfMassList[topRightIdx]
            else:
                pointList[2] = centerOfMassList[topRightIdx]
                pointList[1] = centerOfMassList[bottomLeftIdx]
        pointList[0] = centerOfMassList[topLeftIdx]
        return pointList

def getCenterOfMass(contours):
    pointList = []
    for i in range(len(contours)):
        moment = cv2.moments(contours[i])
        centreOfMassX = int(moment['m10'] / moment['m00'])
        centreOfMassY = int(moment['m01'] / moment['m00'])
        pointList.append([centreOfMassX, centreOfMassY])
    return pointList

def lineAngle(line1, line2):
    return math.acos((line1[0] * line2[0] + line1[1] * line2[1]) / 
       (np.linalg.norm(line1, axis = 0) * np.linalg.norm(line2, axis = 0)))

def selectPatterns(pointList):
    lineList = []
    for i in range(len(pointList)):
        for j in range(i, len(pointList)):
            lineList.append([i, j])
    finalLineList = []
    finalResult = None
    minLengthDiff = -1
    for i in range(len(lineList)):
        for j in range(i, len(lineList)):
            line1 = np.array([pointList[lineList[i][0]][0] -  pointList[lineList[i][1]][0], 
                pointList[lineList[i][0]][1] -  pointList[lineList[i][1]][1]])
            line2 = np.array([pointList[lineList[j][0]][0] -  pointList[lineList[j][1]][0], 
                pointList[lineList[j][0]][1] -  pointList[lineList[j][1]][1]])
            pointIdxList = np.array([lineList[i][0], lineList[i][1], lineList[j][0], lineList[j][1]])
            pointIdxList = np.unique(pointIdxList)
            # print('****')
            if len(pointIdxList) == 3:
                theta = lineAngle(line1, line2)
                if abs(math.pi / 2 - theta) < math.pi / 6:
                    lengthDiff = abs(np.linalg.norm(line1, axis = 0) - np.linalg.norm(line2, axis = 0))
                    if  lengthDiff < minLengthDiff or minLengthDiff < 0:
                        minLengthDiff = abs(np.linalg.norm(line1, axis = 0) - np.linalg.norm(line2, axis = 0))
                        finalResult = pointIdxList

    
    return finalResult

def main():
    path = 'data/1.jpg'

    img = cv2.imread(path)
    show(img)
    thresholdImage, contours, hierarchy = getContours(img)
    img_show = cv2.cvtColor(thresholdImage, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(img_show, contours, -1, (0,255,0), 3)
    show(img_show)

    # qrcode corner has 3 levels
    levelsNum = 3
    patterns, patternsIndices = getContourWithinLevel(levelsNum, contours, hierarchy)
    img_show = cv2.cvtColor(thresholdImage, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(img_show, patterns, -1, (0,255,0), 3)
    show(img_show)
    #in case not all the picture has clear pattern
    while len(patterns) < 3 and levelsNum > 0:
        levelsNum -= 1
        patterns, patternsIndices = getContourWithinLevel(levelsNum, contours, hierarchy)

    interstingPatternList = []
    if len(patterns) < 3 :
        print('no enough pattern')
        return False, []
        # return False

    elif len(patterns) == 3:
        for patternIndex in range(len(patterns)):
            x, y, w, h = cv2.boundingRect(patterns[patternIndex])
            interstingPatternList.append(patterns[patternIndex])

        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        show(img, 'qrcode')
        # return patterns

    elif len(patterns) > 3:
        # sort from small to large
        patternAreaList = np.array(
                [cv2.contourArea(pattern) for pattern in patterns])
        areaIdList = np.argsort(patternAreaList)
        # get patterns without parents
        intrestingPatternIdList = []
        for i in range(len(areaIdList) - 1, 0, -1):
            index = patternsIndices[areaIdList[i]]
            if hierarchy[0][index][3] == -1:
                intrestingPatternIdList.append(index)
            else:
                # We can make sure the parent must appear before chirld because we sorted the list by area
                if not isParentInList(intrestingPatternIdList, index, hierarchy):
                    intrestingPatternIdList.append(index)
        img_show = cv2.cvtColor(thresholdImage, cv2.COLOR_GRAY2BGR)
        for intrestingPatternId in intrestingPatternIdList:
            x, y, w, h = cv2.boundingRect(contours[intrestingPatternId])

            cv2.rectangle(img_show, (x, y), (x + w, y + h), (0, 255, 0), 2)
            interstingPatternList.append(contours[intrestingPatternId])
        show(img_show, 'qrcode')

    img_show = cv2.cvtColor(thresholdImage, cv2.COLOR_GRAY2BGR)
    # cv2.drawContours(img_show, interstingPatternList, -1, (0,255,0), 3)
    # cv2.imwrite('/home/jiangzhiqi/Documents/blog/keefeWu.github.io/source/_posts/opencv实现二维码检测/contours3.jpg', img_show)
    centerOfMassList = getCenterOfMass(interstingPatternList)
    for centerOfMass in centerOfMassList:
        cv2.circle(img_show, tuple(centerOfMass), 3, (0, 255, 0))
    show(img_show, 'qrcode')
    id1, id2, id3 = 0, 1, 2
    if len(patterns) > 3:
        result = selectPatterns(centerOfMassList)
        if result is None:
            print('no correct pattern')
            return False, []
        id1, id2, id3 = result
    interstingPatternList = np.array(interstingPatternList)[[id1, id2, id3]]
    centerOfMassList = np.array(centerOfMassList)[[id1, id2, id3]]
    pointList = getOrientation(interstingPatternList, centerOfMassList)
    img_show = img.copy()
    for point in pointList:
        cv2.circle(img_show, tuple([int(point[0]), int(point[1])]), 10, (0, 255, 0), -1)
    # cv2.imwrite('/home/jiangzhiqi/Documents/blog/keefeWu.github.io/source/_posts/opencv实现二维码检测/result.jpg', img_show)
    point = pointList[0]
    cv2.circle(img_show, tuple([int(point[0]), int(point[1])]), 10, (0, 0, 255), -1)

    show(img_show)
    return True, pointList
    # contours

main()
