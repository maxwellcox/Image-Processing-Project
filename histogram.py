import numpy as np
import math
import matplotlib
import matplotlib.pyplot as plt
import random
from PIL import Image

def image_hist(f, mode):

    imageArray = np.array(f)
    rows, cols = imageArray.shape

    pixelIntensities = np.zeros(256, int)
    histogram = []
    bins = np.array(range(256))

    for row in range(rows):
        for col in range(cols):
            value = imageArray[row][col]
            pixelIntensities[value] += 1        

    if mode != 'u':
        for value in range(256):
            histogram.append(pixelIntensities[value]/(rows*cols))
    else:
        histogram = pixelIntensities
    
    return bins, histogram

def int_xform(f, mode, param):
    imageArray = np.array(f)
    rows, cols = imageArray.shape
    outputArray = np.array(f)

    L = 255
    c = 1.0

    if mode == "negative":
        #s = L - 1 - r
        for row in range(rows):
            for col in range(cols):
                r = imageArray[row][col]
                outputArray[row][col] = (L - 1 - r)

    if mode == "log":
        for row in range(rows):
            for col in range(cols):
                r = imageArray[row][col]
                outputArray[row][col] = int(c*np.log10(1 + r))

    if mode == "gamma":
        for row in range(rows):
            for col in range(cols):
                r = imageArray[row][col]
                outputArray[row][col] = int(c*(r**param))

    return outputArray

def hist_equal(f):

    bins, normalizedHistogram = image_hist(f, 'n')
    newEqualizedHistogram = np.array(range(256))

    for k in range(256):
        summation = 0
        for j in range(k + 1):
            summation += normalizedHistogram[j]
        newEqualizedHistogram[k] == 255*summation

    imageArray = np.array(f)
    rows, cols = imageArray.shape
    outputArray = np.array(f)

    for row in range(rows):
        for col in range(cols):
            r = imageArray[row][col]
            outputArray[row][col] = newEqualizedHistogram[r]

    return outputArray


def main():
    image = Image.open('spillway-dark.tif')
    image.show()

    # bins, histogram = image_hist(image, 'n')
    # plt.bar(bins, histogram)
    # plt.show()

    # imageArray = int_xform(image, 'gamma', 2)
    # newImage = Image.fromarray(imageArray)
    # newImage.show()

    imageArray = hist_equal(image)
    newImage = Image.fromarray(imageArray)
    newImage.show()

    
if __name__ == "__main__":
    main()