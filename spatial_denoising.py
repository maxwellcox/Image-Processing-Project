import numpy as np
import math
import random
import multiprocessing
from threading import *
from PIL import Image

def arith_mean(g, m, n):
    #implements the arithmetic mean filter where g is the input image and m and n are the neighborhood
    #dimensions. This function should return the filtered image as an array. Check
    #your work in your main() function by filtering circuitboard-gaussian.tif as seen in
    #Fig. 5.7(d) in the book.
    originalImageArray = np.array(g)
    oRows, oCols = originalImageArray.shape
    outputImageArray = np.array(g)

    for y in range(oRows):
        for x in range(oCols):
            tempSum = 0
            for r in range(y - int(m/2), y + int(m/2) + 1):
                for c in range(x - int(n/2), x + int(n/2) + 1):
                    if(r < 0 or r >= oRows or c < 0 or c >= oCols):
                        continue
                    tempSum += originalImageArray[r][c]
            outputImageArray[y][x] = tempSum/(m*n)

    return outputImageArray

def geo_mean(g, m, n):
    originalImageArray = np.array(g)
    oRows, oCols = originalImageArray.shape
    outputImageArray = np.array(g)

    for y in range(oRows):
        for x in range(oCols):
            temp = 1.0
            for r in range(y - int(m/2), y + int(m/2) + 1):
                for c in range(x - int(n/2), x + int(n/2) + 1):
                    if(r < 0 or r >= oRows or c < 0 or c >= oCols):
                        continue
                    temp *= originalImageArray[r][c]
            outputImageArray[y][x] = int(temp**(1/(m*n)))

    return outputImageArray

def har_mean(g, m, n):
    originalImageArray = np.array(g)
    oRows, oCols = originalImageArray.shape
    outputImageArray = np.array(g)

    for y in range(oRows):
        for x in range(oCols):
            tempSum = 0
            for r in range(y - int(m/2), y + int(m/2) + 1):
                for c in range(x - int(n/2), x + int(n/2) + 1):
                    if(r < 0 or r >= oRows or c < 0 or c >= oCols):
                        continue
                    tempSum += 1/originalImageArray[r][c]
                    value = m*n
                    value1 = int((m*n)/tempSum)
            outputImageArray[y][x] = int((m*n)/tempSum)

    return outputImageArray

def cthar_mean(g, m, n, q):
    originalImageArray = np.array(g)
    oRows, oCols = originalImageArray.shape
    outputImageArray = np.array(g)

    for y in range(oRows):
        for x in range(oCols):
            if y == 11 and x == 385:
                blah = ''
            tempNumerator = 0
            tempDenominator = 0
            for r in range(y - int(m/2), y + int(m/2) + 1):
                for c in range(x - int(n/2), x + int(n/2) + 1):
                    if(r < 0 or r >= oRows or c < 0 or c >= oCols):
                        continue
                    
                    if(originalImageArray[r][c] == 0 and q == -1.5):
                        tempNumerator += 1
                        tempDenominator += 1
                    else:
                        tempNumerator += (originalImageArray[r][c])**(q + 1)
                        tempDenominator += (originalImageArray[r][c])**q
            
            if(tempDenominator == 0):
                outputImageArray[y][x] = int(tempNumerator)
            else:
                outputImageArray[y][x] = int(tempNumerator/tempDenominator)

    return outputImageArray

def min_filter(g, m, n):
    originalImageArray = np.array(g)
    oRows, oCols = originalImageArray.shape
    outputImageArray = np.array(g)

    for y in range(oRows):
        for x in range(oCols):
            tempMin = 255
            for r in range(y - int(m/2), y + int(m/2) + 1):
                for c in range(x - int(n/2), x + int(n/2) + 1):
                    if(r < 0 or r >= oRows or c < 0 or c >= oCols):
                        continue
                    if tempMin > originalImageArray[r][c]:
                        tempMin = originalImageArray[r][c]
            outputImageArray[y][x] = tempMin

    return outputImageArray

def max_filter(g, m, n):
    originalImageArray = np.array(g)
    oRows, oCols = originalImageArray.shape
    outputImageArray = np.array(g)

    for y in range(oRows):
        for x in range(oCols):
            tempMax = 0
            for r in range(y - int(m/2), y + int(m/2) + 1):
                for c in range(x - int(n/2), x + int(n/2) + 1):
                    if(r < 0 or r >= oRows or c < 0 or c >= oCols):
                        continue
                    if tempMax < originalImageArray[r][c]:
                        tempMax = originalImageArray[r][c]
            outputImageArray[y][x] = tempMax

    return outputImageArray

def median_filter(g, m, n):
    originalImageArray = np.array(g)
    oRows, oCols = originalImageArray.shape
    outputImageArray = np.array(g)

    for y in range(oRows):
        for x in range(oCols):
            intensityValues = []
            for r in range(y - int(m/2), y + int(m/2) + 1):
                for c in range(x - int(n/2), x + int(n/2) + 1):
                    if(r < 0 or r >= oRows or c < 0 or c >= oCols):
                        continue
                    intensityValues.append(originalImageArray[r][c])
            outputImageArray[y][x] = np.median(intensityValues)

    return outputImageArray

def adapt_median_filter(g, s_xy, s_max):
    zmin 

def main():
    #Linear spatial filtering
    image = Image.open('circuitboard-salt.tif')
    image.show()

    outputImageArray = cthar_mean(image, 3, 3, -1.5)
    outputImage = Image.fromarray(outputImageArray)
    outputImage.show()

    #Order-statistic spatial filters
    image = Image.open('hubble.tif')
    image.show()

    outputImageArray = min_filter(image, 15, 15)
    outputImage = Image.fromarray(outputImageArray)
    outputImage.show()



if __name__ == "__main__":
    main()