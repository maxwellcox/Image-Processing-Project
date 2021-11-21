import numpy as np
import math
import random
from PIL import Image

def image_translate(f, tx, ty, mode):
    
    originalImageArray = np.array(f)

    rows, cols = originalImageArray.shape

    backgroundColor = 0

    if mode == "white":
        backgroundColor = 255

    outputArray = np.array([[backgroundColor]*cols]*rows)

    A = np.array([[[1, 0, tx], 
                   [0, 1, ty], 
                   [0, 0, 1]]])

    for row in range(rows):
        for col in range(cols):

            value = originalImageArray[row][col]
            originalPixelPosition = np.array([[col],[row],[1]])
            newPixelPosition = A.dot(originalPixelPosition)

            newX = newPixelPosition[0][0][0]
            newY = newPixelPosition[0][1][0]

            if(newX >= cols or newX < 0 or newY >= rows or newY < 0):
                continue
            outputArray[newY][newX] = value
    return outputArray


def image_scaling(f, cx, cy):

    originalImageArray = np.array(f)
    rows, cols = originalImageArray.shape

    outputArray = np.array([[256]*int(cx*cols)]*int(cy*rows))
    enlargedOriginalImageArray = np.array([[]])

    A = np.array([[[cx, 0, 0],
                   [0, cy, 0], 
                   [0, 0, 1]]])

    #If shrinking the image
    if cx <= 1 and cy <= 1:
        for row in range(int(cy*rows)):
            for col in range(int(cx*cols)):
                
                yScaled = int(row/cy)
                xScaled = int(col/cx)

                value = originalImageArray[yScaled][xScaled]
                originalPixelPosition = np.array([[xScaled],[yScaled],[1]])
                newPixelPosition = A.dot(originalPixelPosition)

                newX = newPixelPosition[0][0][0]
                newY = newPixelPosition[0][1][0]

                if(newX >= cx*cols or newX < 0 or newY >= cy*rows or newY < 0):
                    continue
                outputArray[int(newY)][int(newX)] = value
        return outputArray
    
    #If image is being enlarged vertically
    elif cx <= 1 and cy > 1:
         for row in range(rows):
            for col in range(int(cx*cols)):
                
                xScaled = int(col/cx)

                value = originalImageArray[row][xScaled]
                originalPixelPosition = np.array([[xScaled],[row],[1]])
                newPixelPosition = A.dot(originalPixelPosition)

                newX = newPixelPosition[0][0][0]
                newY = newPixelPosition[0][1][0]

                if(newX >= cx*cols or newX < 0 or newY >= cy*rows or newY < 0):
                    continue
                outputArray[int(newY)][int(newX)] = value

    #If image is being enlarged horizontally
    elif cx > 1 and cy <= 1:
         for row in range(int(cy*rows)):
            for col in range(cols):
                
                yScaled = int(row/cy)

                value = originalImageArray[yScaled][col]
                originalPixelPosition = np.array([[col],[yScaled],[1]])
                newPixelPosition = A.dot(originalPixelPosition)

                newX = newPixelPosition[0][0][0]
                newY = newPixelPosition[0][1][0]

                if(newX >= cx*cols or newX < 0 or newY >= cy*rows or newY < 0):
                    continue
                outputArray[int(newY)][int(newX)] = value
    
    #If image is being enlarged both vertically and horizontally
    elif cx > 1 and cy > 1:
        for row in range(rows):
            for col in range(cols):

                value = originalImageArray[row][col]
                originalPixelPosition = np.array([[col],[row],[1]])
                newPixelPosition = A.dot(originalPixelPosition)

                newX = newPixelPosition[0][0][0]
                newY = newPixelPosition[0][1][0]

                if(newX >= cx*cols or newX < 0 or newY >= cy*rows or newY < 0):
                    continue
                outputArray[int(newY)][int(newX)] = value

    enlargedOriginalImageArray = outputArray

    for row in range(int(cy*rows)):
        for col in range(int(cx*cols)):
            if outputArray[row][col] != 256:
                continue
            else:
                outputArray[row][col] = GetNearestNeighborInterpolation(enlargedOriginalImageArray, row, col)

    return outputArray

def image_shear(f, sv, sh):

    originalImageArray = np.array(f)

    rows, cols = originalImageArray.shape
    additionalPixelsAlongY = 0
    additionalPixelsAlongX = 0

    VHS = np.array([[[1, sh, 0], 
                    [sv, 1, 0], 
                    [0, 0, 1]]])

    originalPixelPosition = np.array([[cols - 1],[0],[1]])
    newPixelPosition = VHS.dot(originalPixelPosition)
    additionalPixelsAlongY = int(newPixelPosition[0][1][0])
    
    originalPixelPosition = np.array([[0],[rows - 1],[1]])
    newPixelPosition = VHS.dot(originalPixelPosition)
    additionalPixelsAlongX = int(newPixelPosition[0][0][0])

    backgroundColor = 0
    outputArray = np.array([[backgroundColor]*(cols + abs(additionalPixelsAlongX))]*(rows + abs(additionalPixelsAlongY)))

    for row in range(rows):
        for col in range(cols):

            value = originalImageArray[row][col]
            originalPixelPosition = np.array([[col],[row],[1]])
            newPixelPosition = VHS.dot(originalPixelPosition)

            newX = int(newPixelPosition[0][0][0])
            newY = int(newPixelPosition[0][1][0])
            
            if additionalPixelsAlongX < 0:
                newX += abs(additionalPixelsAlongX)
            if additionalPixelsAlongY < 0:
                newY += abs(additionalPixelsAlongY)
            outputArray[newY][newX] = value

    return outputArray

def image_rotate(f, theta, mode):
    thetaRad = np.deg2rad(theta)

    originalImageArray = np.array(f)
    originalRows, originalCols = originalImageArray.shape

    radius = np.sqrt(originalRows**2 + originalCols**2)/2

    centerX = originalCols/2
    centerY = originalRows/2

    originalTopLeftTheta = 180 - np.rad2deg(np.arccos(centerX/radius))

    newTopLeftTheta = originalTopLeftTheta + theta

    xPositionOfNewTopLeftCorner = radius*np.cos(np.deg2rad(newTopLeftTheta))
    yPositionOfNewTopLeftCorner = radius*np.sin(np.deg2rad(newTopLeftTheta))

    addX = int(xPositionOfNewTopLeftCorner - (-centerX))
    addY = int(centerY - yPositionOfNewTopLeftCorner)

    if(theta == 180 or theta == 360):
        addX -= 1
        addY -= 1

    addFullX = 0
    addFullY = 0

    newTopLeftLocation = np.array([[]])
    newTopRightLocation = np.array([[]])
    newBottomLeftLocation = np.array([[]])
    newBottomRightLocaiton = np.array([[]])
    outputArray = np.array([[]])
    backgroundColor = 0
    

    R = np.array([[math.cos(-thetaRad), -math.sin(-thetaRad), 0],
                  [math.sin(-thetaRad), math.cos(-thetaRad), 0],
                  [0, 0, 1]])

    if mode == "full" and (theta != 180 or theta != 360):
        xValues = []
        yValues = []
        xTotal = 0
        yTotal = 0

        if(theta == 90 or theta == 270):
            if(originalCols > originalRows):
                yTotal = originalCols
                xTotal = originalCols
                addFullY = (originalCols - originalRows)/2
            if(originalCols < originalRows):
                yTotal = originalRows
                xTotal = originalRows
                addFullX = (originalRows - originalCols)/2

        else:
            originalPixelPosition = np.array([[0],[0],[1]])
            newTopLeftLocation = R.dot(originalPixelPosition)

            originalPixelPosition = np.array([[originalCols - 1],[0],[1]])
            newTopRightLocation = R.dot(originalPixelPosition)
            
            originalPixelPosition = np.array([[0],[originalRows - 1],[1]])
            newBottomLeftLocation = R.dot(originalPixelPosition)

            originalPixelPosition = np.array([[originalCols - 1],[originalRows - 1],[1]])
            newBottomRightLocaiton = R.dot(originalPixelPosition)

            #Check exceeding y values
            if newTopLeftLocation[1] + addY >= originalRows or newTopLeftLocation[1] + addY < 0:
                yValues.append(newTopLeftLocation[1] + addY)
            if newTopRightLocation[1] + addY >= originalRows or newTopRightLocation[1] + addY < 0:
                yValues.append(newTopRightLocation[1] + addY)
            if newBottomLeftLocation[1] + addY >= originalRows or newBottomLeftLocation[1] + addY < 0:
                yValues.append(newBottomLeftLocation[1] + addY)
            if newBottomRightLocaiton[1] + addY >= originalRows or newBottomRightLocaiton[1] + addY < 0:
                yValues.append(newBottomRightLocaiton[1] + addY)

            #Check exceeding x values
            if newTopLeftLocation[0] + addX >= originalCols or newTopLeftLocation[0] + addX < 0:
                xValues.append(newTopLeftLocation[0] + addX)
            if newTopRightLocation[0] + addX >= originalCols or newTopRightLocation[0] + addX < 0:
                xValues.append(newTopRightLocation[0] + addX)
            if newBottomLeftLocation[0] + addX >= originalCols or newBottomLeftLocation[0] + addX < 0:
                xValues.append(newBottomLeftLocation[0] + addX)
            if newBottomRightLocaiton[0] + addX >= originalCols or newBottomRightLocaiton[0] + addX < 0:
                xValues.append(newBottomRightLocaiton[0] + addX)

            for x in xValues:
                if x[0] < 0:
                    addFullX = abs(x[0])
                xTotal += abs(x[0])
            for y in yValues:
                if y[0] < 0:
                    addFullY = abs(y[0])
                yTotal += abs(y[0])
            
            if(xTotal == 0):
                xTotal = originalCols
            if(yTotal == 0):
                yTotal == originalRows

        outputArray = np.array([[backgroundColor]*int(np.ceil(xTotal))]*int(np.ceil(yTotal)))
    else:
        outputArray = np.array([[backgroundColor]*originalCols]*originalRows)

    for row in range(originalRows):
        for col in range(originalCols):

            value = originalImageArray[row][col]
            originalPixelPosition = np.array([[col],[row],[1]])
            newPixelPosition = R.dot(originalPixelPosition)
            
            newX = newPixelPosition[0][0] + addX + addFullX
            newY = newPixelPosition[1][0] + addY + addFullY
            
            if((mode != "full" or theta == 180 or theta == 360) and (newX >= originalCols or newX < 0 or newY >= originalRows or newY < 0)):
                continue

            outputArray[int(newY)][int(newX)] = value

    return outputArray

def GetNearestNeighborInterpolation(array, y, x):

    numRows, numCols = array.shape
    
    xLocations = ['x']
    yLocations = ['y']
    i = 1

    while True:
        xLocations.append('x + ' + str(i))
        xLocations.append('x - ' + str(i))
        yLocations.append('y + ' + str(i))
        yLocations.append('y - ' + str(i))

        random.shuffle(xLocations)
        random.shuffle(yLocations)

        for xloc in xLocations:
            for yloc in yLocations:
                if (xloc.find(str(i)) == -1 and yloc.find(str(i)) == -1):
                    continue
                if eval(xloc) < 0 or eval(xloc) >= numCols:
                    continue
                elif eval(yloc) < 0 or eval(yloc) >= numRows:
                    continue
                value = array[eval(yloc)][eval(xloc)]
                if value != 256 and value != None:
                    return value
                else:
                    continue
        i += 1



def main():
    image = Image.open('girl.tif')
    image.show()
    
    imageArray = image_rotate(image, -45, "crop")
    newImage = Image.fromarray(imageArray)
    newImage.show()
    
if __name__ == "__main__":
    main()
