import numpy as np
import math
import random
import matplotlib.pyplot as plt
import matplotlib.image as img

def conv_2d(f, w):
    #Performs 2d convolution of image f and kernal w. This function should use replicate padding by default
    #Wrtie an image of 512 x 512 pixels that consists of a unit impulse at location (256, 256) and zero elsewhere.
    #The kernel is an array whose size defines the neighborhood of operation, and whose coefficients determine the nature of the filter
    #sum of products of the kernel coefficients and the image pixels encompassed by the kernel
    
    paddedOriginalImageArray = np.array([[]])

    outputImageArray = f
    rows, cols = f.shape

    kernelRows, kernelCols = w.shape

    #Reverse Kernel
    reversedKernelArray = np.rot90(w, 2)

    addedPaddingXTotal = kernelCols - 1
    addedPaddingYTotal = kernelRows - 1

    addedPaddingX = int(addedPaddingXTotal/2)
    addedPaddingY = int(addedPaddingYTotal/2)

    #Apply original image to padded image
    
    #Find distance of outer elements to the center of the original image.
    #Find Center
    centerR = rows/2
    centerC = cols/2
    outerValuesFromCenterOfOriginalImage = []

    centerRSquared = centerR**2
    for c in range(cols):
        colNumber = c - centerC
        result = math.sqrt(colNumber**2 + centerRSquared)
        outerValuesFromCenterOfOriginalImage.append(result)


    #Find the distance between each pixel value to the center of the padded image.

    paddedImageArray = np.array([[256]*(cols+addedPaddingXTotal)]*(rows+addedPaddingYTotal))
    paddedImageRows, paddedImageCols = paddedImageArray.shape

    paddedCenterR = paddedImageRows/2
    paddedCenterC = paddedImageCols/2

    for r in range(paddedImageRows):
        for c in range(paddedImageCols):
            if((r < addedPaddingY or  r >= (rows + addedPaddingY)) or (c < addedPaddingX or c >= (cols + addedPaddingX))):
                distanceFromCenter = math.sqrt((r - paddedCenterR)**2 + (c - paddedCenterC)**2)
                value = 


                # Need to determine replicate padding.
                # What is center of original image.
                continue
            paddedImageArray[r][c] = f[r - addedPaddingY][c - addedPaddingX]


    #Apply Replicate Padding to image
    for r in range(rows + addedPaddingYTotal):
        for c in range(cols + addedPaddingXTotal):
            if((r < addedPaddingY or  r >= rows) or (c < addedPaddingX or c >= cols)):
                paddedOriginalImageArray[r][c] = ApplyReplicatePadding(c, r, paddedOriginalImageArray, reversedKernelArray)
            else:
                continue
    
    
    percentageCompleted = 0
    i = 0
    #Perform convolution
    for y in range(rows + addedPaddingYTotal):
        
        if y == percentageCompleted:
            print("\r", "Convolution percent compeleted: " + str(i) + "%", end="", flush=True)
            percentageCompleted += int((rows + addedPaddingYTotal)/100)
            i += 1

        for x in range(cols + addedPaddingXTotal):
            summation = 0
            if((y < addedPaddingY or  y >= rows) or (x < addedPaddingX or x >= cols)):
                continue
            #Loop through kernel
            for b in range(kernelRows):
                for a in range(kernelCols):
                    s = a - addedPaddingX
                    t = b - addedPaddingY
                    summation += (reversedKernelArray[b][a]*paddedOriginalImageArray[y + t][x + s])
            
            outputImageArray[y - addedPaddingY][x - addedPaddingX] = summation

    print("[COMPLETED]")
    return outputImageArray    

def gauss_kernel(m, sig, K=1):
    outputKernel = np.array([[0]*m]*m)

    for t in range(m):
        for s in range(m):
            r = np.sqrt(s**2 + t**2)
            outputKernel[t][s] = K*(np.exp((0 - ((r**2)/(2*sig**2)))))
    return outputKernel

def ApplyReplicatePadding(x, y, paddedImageArray, w):

    kernelNumRows, kernelNumCols = w.shape
    paddedNumRows, paddedNumCols = paddedImageArray.shape
    
    xLocations = ['x']
    yLocations = ['y']

    rangeValue = int((kernelNumCols - 1)/2) + 1
    for i in range(1, rangeValue):
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
                if eval(xloc) < 0 or eval(xloc) >= paddedNumCols:
                    continue
                elif eval(yloc) < 0 or eval(yloc) >= paddedNumRows:
                    continue
                value = paddedImageArray[eval(yloc)][eval(xloc)]
                if value != 256 and value != None:
                    return value
                else:
                    continue
        i += 1

def main():
    
    #2D convolution.
    #After testing it the 2d convolution did work as I was expecting it to.

    #Testing
    array = np.array([[1,2,3],
                      [4,5,6],
                      [7,8,9]])

    print(array)
    rotatedArray = np.rot90(array, 2)

    print(rotatedArray)

    #Low Pass Filtering
    image = img.imread('testpattern1024.tif')
    plt.imshow(image, cmap='gray')
    plt.title("Original testPattern1024")
    plt.show()

    w = gauss_kernel(180, 30)

    newImage = conv_2d(image, w)
    plt.imshow(newImage, cmap='gray')
    plt.title("Convolved testPattern1024 with Gaussian Kernel")
    plt.show()

    #Sharpening
    #1.
    image = Image.open('blurry-moon.tif')
    image.show()
    w = gauss_kernel(31, 5)
    blurredImageArray = conv_2d(image, w)
    maskedImageArray = np.subtract(np.array(image),blurredImageArray)
    finalImageArray = np.add(np.array(image), maskedImageArray)
    newImage = Image.fromarray(finalImageArray)
    newImage.show()

    #2.
    image = Image.open('blurry-moon.tif')
    image.show()
    w = gauss_kernel(31, 5)
    k = 2
    blurredImageArray = conv_2d(image, w)
    maskedImageArray = np.subtract(np.array(image),blurredImageArray)
    finalImageArray = np.add(np.array(image), k*maskedImageArray)
    newImage = Image.fromarray(finalImageArray)
    newImage.show()

    #3.
    image = Image.open('blurry-moon.tif')
    image.show()

    c = -1
    w = np.array([[0, 1, 0],
                  [1, -4, 1],
                  [0, 1, 0]])
    blurredImageArray = conv_2d(image, w)
    finalImageArray = np.add(np.array(image), blurredImageArray * c)
    newImage = Image.fromarray(finalImageArray)
    newImage.show()

    #4.
    image = Image.open('blurry-moon.tif')
    image.show()

    c = -1
    w = np.array([[-1, -2, -1],
                  [0, 0, 0],
                  [1, 2, 1]])
    blurredImageArray = conv_2d(image, w)
    finalImageArray = np.add(np.array(image), blurredImageArray * c)
    newImage = Image.fromarray(finalImageArray)
    newImage.show()

    #I think the best one to my knowledge would probably be the Laplacian. I didn't quite get the result I wanted from it but I think that one was probably the best of the 4.

if __name__ == "__main__":
    main()