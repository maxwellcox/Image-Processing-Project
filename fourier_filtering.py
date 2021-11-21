import numpy as np
import math
import random
from PIL import Image

def dft_filtering(f, H, padmode, scaling):
    #filters image f with a given filter transfer function H.
    fArray = np.array(f)

    #1. Obtain padding sizes
    M, N = fArray.shape
    P = 2*M
    Q = 2*N

    #2. Form a padded image fp of size PxQ using zero or replicate padding
    print('\nApplying Padding')

    if padmode == 'zeros':
        fp = np.array([[0.0]*Q]*P)
    else:
        fp = np.array([[256.0]*Q]*P)

    for r in range(P):
        for c in range(Q):
            if((r < P/4 or  r >= P/4 + M) or (c < Q/4 or c >= Q/4 + N)):
                continue
            else:
                fp[r][c] = float(fArray[r - int(P/4)][c - int(Q/4)])
    
    if(padmode != 'zeros'):
        for r in range(P):
            for c in range(Q):
                if((r < P/4 or  r >= P/4 + M) or (c < Q/4 or c >= Q/4 + N)):
                    fp[r][c] = ApplyReplicatePadding(c, r, fp)
    
    #3. Multiply fp BY (-1)^(x + y) to center the Fourier transform on the PxQ freqency rectangle
    fp = minus_one(fp)

    #4. Compute the DFT, Fuv, of the image from step 3
    print('Computing the DFT of the image')
    Fuv = dft2d(fp)

    #5. Construct a real, symmetric filter transfer function, Huv, of size PxQ with center at P/2,Q/2
    #Done use H which is given (Need to make sure that H and fp are the same size)

    #6 Form the product Guv using elementwise multiplication
    
    Guv = np.array(Fuv)

    for i in range(P):
        for k in range(Q):
            Guv[i][k] = H[i][k]*Fuv[i][k]
        
    #7.Obtain the filtered image (of size P x Q) by computing the IDFT of Guv
    print('Computing the IDFT of the processed image')
    gp = minus_one(idft2D(Guv))

    #8 Obtain the final filtered result, g, of the same size as the input image by extracting the M x N region from the top, left quadrant of gp
    print('Aquiring final image')
    g = np.array(f)

    for y in range(M):
        for x in range(N):
            g[y][x] = gp[y][x]
    
    return g


def lp_filter_tf(type, P, Q, D_0, n=2):
    #Generates a P * Q low pass filter transfer function H

    H = np.array([[0.0]*(Q)]*(P))

    if type == 'ideal':
        for v in range(P):
            for u in range(Q):
                Duv = math.sqrt((u - P/2)**2 + (v - Q/2)**2)
                if Duv <= D_0:
                    H[v][u] = 1.0

    if type == 'gaussian':
        for v in range(P):
            for u in range(Q):
                Duv = math.sqrt((u - P/2)**2 + (v - Q/2)**2)
                H[v][u] = np.exp((-(Duv**2)/(2*D_0**2)))

    if type == 'butterworth':
        for v in range(P):
            for u in range(Q):
                Duv = math.sqrt((u - P/2)**2 + (v - Q/2)**2)
                H[v][u] = 1/(1 + (Duv/D_0)**(2*n))
    
    return H

def hp_filter_tf(type, P, Q, D_0, n=2):
    #Generates a P * Q high pass filter transfer function H

    H = np.array([[0.0]*(Q)]*(P))

    if type == 'ideal':
        for v in range(P):
            for u in range(Q):
                Duv = math.sqrt((u - P/2)**2 + (v - Q/2)**2)
                if Duv > D_0:
                    H[v][u] = 1.0

    if type == 'gaussian':
        for v in range(P):
            for u in range(Q):
                Duv = math.sqrt((u - P/2)**2 + (v - Q/2)**2)
                H[v][u] = (1 - np.exp((-(Duv**2)/(2*D_0**2))))

    if type == 'butterworth':
        for v in range(P):
            for u in range(Q):
                Duv = math.sqrt((u - P/2)**2 + (v - Q/2)**2)
                if(Duv == 0.0):
                    H[v][u] = 0.0
                else:
                    H[v][u] = 1/(1 + (D_0/Duv)**(2*n))
    
    return H

def minus_one(f):
    #takes your image f as an input and multiplies it by (-1)^(x+y) to return g. the Input image should be floating point, so your function should check that

    originalImageArray = np.array(f)
    outputImageArray = np.array(f)
    rows, cols = originalImageArray.shape

    if originalImageArray.dtype != float:
        print("Image does not contain floating point numbers.")
    for r in range(rows):
        for c in range(cols):
            outputImageArray[r][c] = originalImageArray[r][c]*(-1)**(c + r)

    return outputImageArray


def dft2d(f):
    #takes your image f and computes a 2d FFT and returns F. Using the np.fft.fft()
    percentageCompleted = 0
    i = 0   

    fArray = np.array(f)
    M, N = fArray.shape
    F = np.array([[0.+0.j]*N]*M, dtype=np.complex)
    
    i = 0
    for y in range(M):
        if y == percentageCompleted:
            print("\r", str(i) + "%", end="", flush=True)
            percentageCompleted += int(M/100)
            i += 1

        rowFourierTransform = np.fft.fft(fArray[y]) # will perform the fast fourier transform for each row.
        for index in range(len(rowFourierTransform)):
            tempSum = 0
            for v in range(M):
                tempSum += rowFourierTransform[index]*np.exp((-2j*np.pi*v*y)/M)
            F[y][index] = tempSum

    return F

def idft2D(F):
    FArray = np.array(F)
    f = np.array(F)

    M,N = FArray.shape

    for v in range(M):
        rowFourierTransform = np.fft.fft(np.conjugate(FArray[v]))
        for index in range(len(rowFourierTransform)):
            tempSum = 0
            for y in range(M):
                tempSum += rowFourierTransform[index]*np.exp((2j*np.pi*y*v)/M)
            f[v][index] = tempSum/(M*N)

    return f

def displayFilter(H):
    filterToDisplay = H
    P, Q = H.shape

    for v in range(P):
        for u in range(Q):
            filterToDisplay[v][u] = H[v][u]*255
    image = Image.fromarray(filterToDisplay)
    image.show()

def ApplyReplicatePadding(x, y, paddedImageArray):

    paddedNumRows, paddedNumCols = paddedImageArray.shape
    
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
    # image = Image.open('testpattern1024.tif')
    # image.show()

    #Low pass filtering
    # idealLowPassFilter = lp_filter_tf('ideal', 512, 512, 120)
    # gaussianLowPassFilter = lp_filter_tf('gaussian', 512, 512, 120)
    # butterworthLowPassFilter = lp_filter_tf('butterworth', 512, 512, 120)
    # displayFilter(idealLowPassFilter)
    # displayFilter(gaussianLowPassFilter)
    # displayFilter(butterworthLowPassFilter)

    # #High pass filtering
    # idealHighPassFilter = hp_filter_tf('ideal', 512, 512, 120)
    # gaussianHighPassFilter = hp_filter_tf('gaussian', 512, 512, 120)
    # butterworthHighPassFilter = hp_filter_tf('butterworth', 512, 512, 120)
    # displayFilter(idealHighPassFilter)
    # displayFilter(gaussianHighPassFilter)
    # displayFilter(butterworthHighPassFilter)

    #Low pass filtering
    image = Image.open('testpattern1024.tif')
    image.show()

    P, Q = np.array(image).shape
    H = lp_filter_tf('gaussian', P, Q, 120) #type, P, Q, D_0, n=2)
    g = dft_filtering(image, H, 'zeros', 0)
    image = Image.fromarray(g)
    image.show()

    H = lp_filter_tf('butterworth', P, Q, 72) #type, P, Q, D_0, n=2)
    g = dft_filtering(image, H, 'zeros', 0)
    image = Image.fromarray(g)
    image.show()

    #Image Sharpening
    image = Image.open('blurry-moon.tif')
    image.show()

    H = lp_filter_tf('butterworth', P, Q, 72) #type, P, Q, D_0, n=2)
    g = dft_filtering(image, H, 'zeros', 0)
    gmask = np.array(image) - g

    g = np.array(image) + g
    image = Image.fromarray(g)
    image.show()

    gmask = np.array(image) - g

    k = 1.5
    g = np.array(image) + k*g
    image = Image.fromarray(g)
    image.show()

if __name__ == "__main__":
    main()