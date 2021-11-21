import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.image as img
from numpy.core.fromnumeric import transpose
from numpy.core.shape_base import block

def performHMT(I,B):
    bRows, bCols = B.shape

    m = int(bRows/2)
    n = int(bCols/2)

    iRows = 1
    iCols = 1
    try:
        I = I[:,:,0]
    except IndexError:
        variable = True #This is just used to catch the Index Error when it occurs.
    
    iRows, iCols = I.shape

    paddedImage = np.array([[0]*(2*n + iCols)]*(2*m + iRows))
    paddedImage[m:iRows+m,n:iCols+n] = I
    paddedImageRows, paddedImageCols = paddedImage.shape

    HMTImage = np.array([[0]*(2*n + iCols)]*(2*m + iRows))
        
    bPatterns = CreateDontCareStructuralElements(B)

    for r in range(paddedImageRows):
        for c in range(paddedImageCols):
            
            rIndex = r + m
            cIndex = c + n

            if(rIndex >= paddedImageRows or cIndex >= paddedImageCols):
                continue

            A = paddedImage[r:r+bRows,c:c+bCols]
            
            for b in bPatterns:
                b = b*255
                if(np.array_equal(A,b)):
                    HMTImage[rIndex][cIndex] = 255
                    break

    outputImage = HMTImage[m:iRows+m,n:iCols+n]
    
    return outputImage

def CreateDontCareStructuralElements(B):
    bPatterns = []
    dontCareCount = int(np.sum(B == -1))
    bPatterns.append(B)
    if(dontCareCount > 0):
        for i in range(dontCareCount):
            bSubList = []
            for bP in bPatterns:
                #Find index of first nan
                index = []
                tempIndex = np.where(bP == -1)
                index = np.array([tempIndex[0][0], tempIndex[1][0]])
                for x in range(2):
                    bTemp = np.array([[0]*3]*3)
                    bTemp[:,:] = bP
                    bTemp[index[0]][index[1]] = x
                    bSubList.append(bTemp)
            bPatterns = bSubList
    
    return bPatterns

def morpho_erode(I, B, padval):
    
    bRows = 1
    bCols = 1
    try:
        bRows, bCols = B.shape
    except ValueError:
        bRows = B.shape[0]

    m = int(bRows/2)
    n = int(bCols/2)

    iRows = 1
    iCols = 1
    try:
        I = I[:,:,0]
    except IndexError:
        variable = True #This is just used to catch the Index Error when it occurs.
    
    iRows, iCols = I.shape

    outputImage= np.array([[0]*iCols]*iRows)
    outputImage[:,:] = I

    paddedImage = np.array([[padval]*(2*n + iCols)]*(2*m + iRows))
    paddedImage[m:iRows+m,n:iCols+n] = I

    expectedValue = bRows*bCols
    
    paddedImageRows, paddedImageCols = paddedImage.shape

    for r in range(paddedImageRows):
        for c in range(paddedImageCols):
            
            A = paddedImage[r:r+bRows,c:c+bCols]
            aRows, aCols = A.shape

            rIndex = r + m
            cIndex = c + n
            if(rIndex >= iRows):
                rIndex = r
            if(cIndex >= iCols):
                cIndex = c

            if((aRows != bRows or aCols != bCols) or rIndex == iRows or cIndex == iCols):
                continue
            d = A*B

            validPixelCount = (d == 255).sum()

            if(validPixelCount == expectedValue):
                outputImage[rIndex][cIndex] = 255
            else:
                outputImage[rIndex][cIndex] = 0
    
    return outputImage

def morpho_dilate(I, B, padval):
    
    Br = np.rot90(B, 2)
    filledBr = Br*255
    
    bRows = 1
    bCols = 1
    try:
        bRows, bCols = Br.shape
    except ValueError:
        bRows = Br.shape[0]

    m = int(bRows/2)
    n = int(bCols/2)

    iRows = 1
    iCols = 1
    try:
        I = I[:,:,0]
    except IndexError:
        variable = True #This is just used to catch the Index Error when it occurs.
    
    iRows, iCols = I.shape
    paddedImage = np.array([[padval]*(2*n + iCols)]*(2*m + iRows))
    paddedImageRows, paddedImageCols = paddedImage.shape
    paddedImage[m:iRows+m,n:iCols+n] = I
    outputPaddedImage = np.array([[padval]*(2*n + iCols)]*(2*m + iRows))
    
    for r in range(paddedImageRows):
        for c in range(paddedImageCols):
            rIndex = r + m
            cIndex = c + n

            if(rIndex >= iRows or cIndex >= iCols or (paddedImage[rIndex][cIndex] != 255)):
                continue
            else:
                outputPaddedImage[r:r+bRows,c:c+bCols] = filledBr
    
    return outputPaddedImage

def morpho_open(I, B):

    try:
        I = I[:,:,0]
    except IndexError:
        variable = True #This is just used to catch the Index Error when it occurs.

    erodedImage = morpho_erode(I,B,1)
    openedImage = morpho_dilate(erodedImage,B,1)

    return openedImage

def morpho_close(I, B):

    try:
        I = I[:,:,0]
    except IndexError:
        variable = True #This is just used to catch the Index Error when it occurs.

    dilatedImage = morpho_dilate(I,B,1)
    openedImage = morpho_erode(dilatedImage,B,1)

    return openedImage

def morpho_boundary(I,B):
    try:
        I = I[:,:,0]
    except IndexError:
        variable = True #This is just used to catch the Index Error when it occurs.
    
    erodedImage = morpho_erode(I,B,1)
    boundary = I - erodedImage
    boundary = np.clip(boundary,0,255)
    
    return boundary

def morpho_thin(I, B, numiter):

    bRows = bCols = 1

    for b in B:
        br = bc = 1
        try:
            br, bc = b.shape
        except ValueError:
            br = b.shape[0]
        
        if br > bRows:
            bRows = br
        if bc > bCols:
            bCols = bc

    m = int(bRows/2)
    n = int(bCols/2)

    iRows = 1
    iCols = 1
    try:
        I = I[:,:,0]
    except IndexError:
        variable = True #This is just used to catch the Index Error when it occurs.
    
    iRows, iCols = I.shape

    paddedImage = np.array([[0]*(2*n + iCols)]*(2*m + iRows))
    paddedImage[m:iRows+m,n:iCols+n] = I
    
    paddedImageRows, paddedImageCols = paddedImage.shape

    for k in range(numiter):
        HMTImage = np.array([[0]*(2*n + iCols)]*(2*m + iRows))
        
        bPatterns = CreateDontCareStructuralElements(B[k])

        for r in range(paddedImageRows):
            for c in range(paddedImageCols):
                
                if(r == 4 and c == 1 and k == 2):
                    blah = ''
                rIndex = r + m
                cIndex = c + n

                if(rIndex >= paddedImageRows or cIndex >= paddedImageCols):
                    continue

                A = paddedImage[r:r+bRows,c:c+bCols]
                
                for b in bPatterns:
                    b = b*255
                    if(np.array_equal(A,b)):
                        HMTImage[rIndex][cIndex] = 255
                        break

        paddedImage = paddedImage - HMTImage
        paddedImage = np.clip(paddedImage,0,255)

    outputImage = paddedImage[m:iRows+m,n:iCols+n]
    
    return outputImage

def morpho_prun(I, numthin, numdil):
    
    iRows = 1
    iCols = 1
    try:
        I = I[:,:,0]
    except IndexError:
        variable = True #This is just used to catch the Index Error when it occurs.
    
    iRows, iCols = I.shape
    x = -1
    B = np.array([[[x,0,0],
                   [1,1,0],
                   [x,0,0]],

                  [[x,1,x],
                   [0,1,0],
                   [0,0,0]],

                  [[0,0,x],
                   [0,1,1],
                   [0,0,x]],
                  
                  [[0,0,0],
                   [0,1,0],
                   [x,1,x]],
                  
                  [[1,0,0],
                   [0,1,0],
                   [0,0,0]],
                  
                  [[0,0,1],
                   [0,1,0],
                   [0,0,0]],
                  
                  [[0,0,0],
                   [0,1,0],
                   [0,0,1]],
                  
                  [[0,0,0],
                   [0,1,0],
                   [1,0,0]]])
    
    X1 = morpho_thin(I, B, 8)
    for a in range(numthin - 1):
        X1 = morpho_thin(X1, B, 8)
    
    X2 = np.array([[0]*iCols]*iRows)
    for k in range(len(B)):
        X2 = X2 + performHMT(X1, B[k])

    H = np.array([[1,1,1],
                  [1,1,1],
                  [1,1,1]])

    X3 = np.array([[0]*iCols]*iRows)
    X3[:,:] = X2
    for a in range(numdil):
        hRows,hCols = H.shape
        m = int(hRows/2)
        n = int(hCols/2)
        unpaddedDilatedImage = np.array([[0]*(iCols)]*(iRows))
        dilatedImage = morpho_dilate(X3,H,0)
        unpaddedDilatedImage = dilatedImage[m:iRows+m,n:iCols+n]
        X3 = unpaddedDilatedImage & I
    
    X4 = X1 | X3

    return X4

def morpho_geo_dilate(F,G,B):

    D = []
    D.append(F)

    n = 0
    while True:
        n += 1
        Dn = morpho_dilate(D[n-1],B,0) & G
        if(np.array_equal(Dn,G)):
            return Dn
        else:
            D.append(Dn)

def morpho_geo_erod(F, G, B, n):
    E = []
    E.append(F)

    for x in range(1,n):
        En = morpho_erode(E[x-1],B,0) | G
        if(np.array_equal(En,G)):
            return En
        else:
            E.append(En)

def main():
    B = np.array([[1,1,1],
                  [1,1,1],
                  [1,1,1]])

    image = img.imread('UTK.tif')
    
    #Erosion
    erodedImage = morpho_erode(image, B, 1)
    erodedImage = morpho_erode(erodedImage, B, 1)
    erodedImage = morpho_erode(erodedImage, B, 1)
    fig = plt.figure()
    ax = fig.add_subplot(1, 2, 1)
    plt.imshow(image, cmap='gray')
    ax.set_title('Original Image')
    ax = fig.add_subplot(1,2,2)
    plt.imshow(erodedImage, cmap='gray')
    ax.set_title('Eroded Image')
    plt.show()

    # #Dilation
    dilatedImage = morpho_dilate(image, B, 1)
    dilatedImage = morpho_dilate(dilatedImage, B, 1)
    dilatedImage = morpho_dilate(dilatedImage, B, 1)
    fig = plt.figure()
    ax = fig.add_subplot(1, 2, 1)
    plt.imshow(image, cmap='gray')
    ax.set_title('Original Image')
    ax = fig.add_subplot(1,2,2)
    plt.imshow(dilatedImage, cmap='gray')
    ax.set_title('Dilated Image')
    plt.show()

    # #Opening and Closing
    image = img.imread('circuitmask.tif')
    B = np.ones((28,28))
    openImage = morpho_open(image, B)
    closedImage = morpho_close(image, B)
    
    fig = plt.figure()
    ax = fig.add_subplot(2, 2, 1)
    plt.imshow(image, cmap='gray')
    ax.set_title('Original Image')

    ax = fig.add_subplot(2,2,2)
    plt.imshow(openImage, cmap='gray')
    ax.set_title('Open Image')
    
    ax = fig.add_subplot(2, 2, 3)
    plt.imshow(image, cmap='gray')
    ax.set_title('Original Image')

    ax = fig.add_subplot(2, 2, 4)
    plt.imshow(closedImage, cmap='gray')
    ax.set_title('Closed Image')
    plt.show()

    # #Boundary extraction
    B = np.ones((3,3))
    image = img.imread('testpattern512-binary.tif')
    boundaryImage = morpho_boundary(image, B)
    fig = plt.figure()
    ax = fig.add_subplot(1, 2, 1)
    plt.imshow(image, cmap='gray')
    ax.set_title('Original Image')
    ax = fig.add_subplot(1,2,2)
    plt.imshow(boundaryImage, cmap='gray')
    ax.set_title('Image Boundary')
    plt.show()

    #Thinning
    image = img.imread('UTK.tif')
    B = np.array([ [[0,0,0],
                    [0,1,1],
                    [0,1,1]],
                    
                    [[0,1,1],
                     [0,1,1],
                     [0,1,1]],

                            ])
    
    numiter,a,b = B.shape
    thinImage = morpho_thin(image, B, numiter)

    fig = plt.figure()
    ax = fig.add_subplot(1, 2, 1)
    plt.imshow(image, cmap='gray')
    ax.set_title('Original Image')
    ax = fig.add_subplot(1,2,2)
    plt.imshow(thinImage, cmap='gray')
    ax.set_title('Thin Image')
    plt.show()

    prunImage = morpho_prun(thinImage, 3, 3)

    fig = plt.figure()
    ax = fig.add_subplot(1, 2, 1)
    plt.imshow(thinImage, cmap='gray')
    ax.set_title('Thin Image')
    ax = fig.add_subplot(1,2,2)
    plt.imshow(prunImage, cmap='gray')
    ax.set_title('Pruned Image')
    plt.show()

    #Geodesic dilation and erosion
    aIRow = aICol = 1
    arrowImage = img.imread('calculator-binary-upper-arrow.tif')
    try:
        arrowImage = arrowImage[:,:,0]
    except IndexError:
        variable = True #This is just used to catch the Index Error when it occurs.
    aIRow, aICol = arrowImage.shape

    iRow = iCol = 1
    image = img.imread('calculator-binary.tif')
    try:
        image = image[:,:,0]
    except IndexError:
        variable = True #This is just used to catch the Index Error when it occurs.
    
    
    iRow, iCol = image.shape
    outputImage = np.array([[0]*iCol]*iRow)
    outputImage[:,:] = image
    arrowImageFill = np.array([[0]*aICol]*aIRow)

    matchFound = False
    for r in range(iRow):
        if matchFound:
            break
        for c in range(iCol):
            neighborhood = image[r:aIRow+r,c:aICol+c]
            if(np.array_equal(neighborhood,arrowImage)):
                outputImage[r:aIRow+r,c:aICol+c] = arrowImageFill
                matchFound = True
                break
    
    fig = plt.figure()
    ax = fig.add_subplot(1, 2, 1)
    plt.imshow(image, cmap='gray')
    ax.set_title('Calculator Image')
    ax = fig.add_subplot(1,2,2)
    plt.imshow(outputImage, cmap='gray')
    ax.set_title('Output Calculator Image')
    plt.show()


if __name__ == "__main__":
    main()