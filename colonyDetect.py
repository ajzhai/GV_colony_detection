###############################################################################
# GV Colony Detection and Instance Segmentation
# Albert Zhai, March 2019
#
# This program will detect GV-expressing colonies in a collapse xAM image file
# specified by the user and provide a segmentation mask where each
# colony's region has a unique pixel value.  The input image should be a
# grayscale collapse xAM-dB image where a pixel value of 0 corresponds to
# -70 dB and a pixel value of 255 corresponds to 0 dB. A second image of
# the same size can also be specified as the background for final
# visualization.
# The program will display windows showing the segmentation results, one for
# each tuple of filter parameters defined at the top of the file. The user
# should type the number of the window which has the best results, and then
# a final visualization will be displayed in a window and outputs will
# be saved.
###############################################################################

import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import sys


MIN_CIRCULARITY = 0.25

# Create set of heuristic filter parameters to try
heurs = []
for highpass in [100, 110, 120]:  # High-pass intensity thresholds
    for erodesize in [None, (3, 3), (5, 5)]:  # Erosion kernel (x, y) sizes
        # High-pass threshold, low-pass threshold, min area, max area, blur size, erosion size
        heurs.append((highpass, 250, 80, 700, 5, erodesize))


def colonyContours(img, minVal, maxVal, minSize, maxSize, blurSize, blurMode=1, erodeKer=None):
    """
    Finds contours of bacterial colonies in image, filtering with
    the given parameters. Returns copy of source image with contours
    drawn in red.
    :param img: The source image array (should be grayscale)
    :param minVal: Minimum pixel intensity threshold
    :param maxVal: Maximum pixel intensity
    :param minSize: Minimum contour area to be considered a possible colony
    :param maxSize: Maximum contour area to be considered
    :param blurSize: The vertical size of the Gaussian blur kernel (must be odd integer)
    :param blurMode: Gaussian blur if 1, median blur if 0
    :param erodeKer: Size of kernel if want to erode
    :return: List of detected colony contours
    """

    # Apply low-pass intensity filter
    ret, imgTrunc = cv.threshold(img, maxVal, 255, cv.THRESH_TOZERO_INV)

    # Apply high-pass intensity filter
    ret, imgTrunc = cv.threshold(imgTrunc, minVal, 255, cv.THRESH_TOZERO)

    # Slightly blur the image to smooth out small noise
    if blurMode == 1:
        imBlur = cv.GaussianBlur(imgTrunc, (1, blurSize), 0)
    else:
        imBlur = vMedBlur(imgTrunc, blurSize)

    # Apply high-pass filter again after blur
    ret, thresh = cv.threshold(imBlur, minVal, 255, cv.THRESH_TOZERO)

    # Find contours and fill in areas with white. We do this first to prevent interior
    # holes from being widened in erosion.
    _, contours, hierarchy = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    cv.drawContours(thresh, contours, -1, 255, thickness=cv.FILLED)

    # Apply erosion and then dilation
    if erodeKer:
        kernel = np.ones(erodeKer, np.uint8)
        imErode = cv.erode(thresh, kernel, iterations=1)
        thresh = cv.dilate(imErode, kernel, iterations=1)

    # Get final contours.
    _, contours, hierarchy = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    colonies = []
    for contour in contours:
        # Calculate circularity measure using contour area and minimum enclosing circle area
        _, radius = cv.minEnclosingCircle(contour)
        cntArea = cv.contourArea(contour)
        circularity = cntArea / (np.pi * radius ** 2)

        if cntArea < minSize or cntArea > maxSize or circularity < MIN_CIRCULARITY:
            # Size is clearly too small or large or shape is too thin
            continue
        colonies.append(contour)
    return colonies


def drawColonies(img, contours, background=None, labelType='id', scale=1):
    """
    Draws given colony contours onto copy of image with optional informative labels.
    :param img: The source image array (should be grayscale)
    :param contours: List of colony contours
    :param background: The background image array to draw on
    :param labelType: What property to label each contour with. Options are 'id' (default),
                      'inten', 'area', 'circ', 'none'
    :param scale: Integer factor to scale the output image by (larger images have clearer text)
    :return: Copy of source image array with contours (and perhaps more) drawn on.
    """

    # Draw on original if no background specified
    if background is None:
        background = cv.cvtColor(img, cv.COLOR_GRAY2BGR)

    imContours = cv.resize(background, (scale * img.shape[1], scale * img.shape[0]))

    for i, contour in enumerate(contours):
        # Calculate values to label with
        _, radius = cv.minEnclosingCircle(contour)
        cntArea = cv.contourArea(contour)
        circularity = cntArea / (np.pi * radius ** 2)

        # Draw contours in red
        cv.drawContours(imContours, [contour * scale], 0, (0, 0, 255), 1)

        textOrg = (contour[0, 0, 0] * scale, contour[0, 0, 1] * scale)  # Top-most point
        fontScale = 0.7
        if labelType == 'id':
            cv.putText(imContours, '%d' % i,  # IDs increase with distance from bottom of image
                       textOrg, cv.FONT_HERSHEY_PLAIN, fontScale, (0, 255, 0))
        elif labelType == 'inten':
            cv.putText(imContours, '%.1f' % contourIntensity(img, contour),
                       textOrg, cv.FONT_HERSHEY_PLAIN, fontScale, (0, 255, 0))
        elif labelType == 'area':
            cv.putText(imContours, str(cntArea),
                       textOrg, cv.FONT_HERSHEY_PLAIN, fontScale, (0, 255, 0))
        elif labelType == 'circ':
            cv.putText(imContours, '%.2f' % circularity,
                       textOrg, cv.FONT_HERSHEY_PLAIN, fontScale, (0, 255, 0))
    return imContours


def contourIntensity(img, contour):
    """
    Returns average pixel intensity of the image inside the given contour.
    :param img: The source image array (should be grayscale)
    :param contour: The contour to find the average intensity within
    :return: The average intensity inside the contour
    """
    mask = np.zeros(img.shape)
    cv.drawContours(mask, [contour], 0, 255, thickness=cv.FILLED)
    intens = []
    for y in range(img.shape[0]):
        for x in range(img.shape[1]):
            if mask[y][x]:
                intens.append(img[y][x])
    return np.mean(intens)


def segMask(img, contours):
    """
    Returns segmentation mask of the colonies in the image. Assigns IDs to each contour
    increasing with distance from the bottom of the image. Pixel values of the mask
    are as follows:
    For IDs 0-255, RGB = (ID, 0, 0).
    For IDs 256-511, RGB = (255, ID - 256, 0)
    For IDs 512-767, RGB = (255, 255, ID - 512)
    Cannot handle more than 768 colonies.
    :param img: The source image array (only used for dimensions)
    :param contours: The contours of the individual colonies
    :return: RGB segmentation mask image array
    """
    mask = np.zeros((img.shape[0], img.shape[1], 3))
    for i in range(len(contours)):
        if i > 511:
            cv.drawContours(mask, contours, i, (i - 512, 255, 255), thickness=cv.FILLED)
        elif i > 255:
            cv.drawContours(mask, contours, i, (0, i - 256, 255), thickness=cv.FILLED)
        else:
            cv.drawContours(mask, contours, i, (0, 0, i), thickness=cv.FILLED)
    return mask


def showOptions(img):
    """Displays all heuristic filter parameter options and gets user's choice."""
    for i, filterParams in enumerate(heurs):
        colContours = colonyContours(img, filterParams[0], filterParams[1],
                                     filterParams[2], filterParams[3], filterParams[4],
                                     erodeKer=filterParams[5])
        imContours = drawColonies(img, colContours, labelType='', scale=1)
        cv.imshow('Option %d: High-pass threshold %d, Erosion size %s' %
                  (i + 1, filterParams[0], str(filterParams[5])), imContours)
    choice = 0
    while choice < 49 or choice > 57:  # 49 is ASCII code for '1'
        print('Enter best choice (1-9): ', end='')
        choice = cv.waitKey(0)
        print(choice - 48)
    cv.destroyAllWindows()
    return choice - 49


def avgIntenHistogram(img, contours, outFile):
    """
    CURRENTLY UNUSED: Saves a histogram of the average intensities of the colonies
    in the image.
    """
    avgIntens = []
    for contour in contours:
        avgIntens.append(contourIntensity(img, contour))
    plt.hist(avgIntens, edgecolor='black')
    plt.title('Colony Average Intensities')
    plt.xlabel('Average Pixel Intensity (out of 255)')
    plt.ylabel('Frequency')
    plt.savefig(outFile)
    plt.close()


def colonyAnalysis(inFile, bgFile=None):
    """Performs all the major steps of colony intensity analysis on the given image file."""
    im = cv.imread(inFile, 0)
    # Use user-chosen filtering parameters
    filterParams = heurs[showOptions(im)]
    colContours = colonyContours(im, filterParams[0], filterParams[1],
                                 filterParams[2], filterParams[3], filterParams[4],
                                 erodeKer=filterParams[5])

    # Do not enlarge the visualization if image is too big already
    if im.shape[1] > 1000 or im.shape[0] > 600:
        scale = 1
    else:
        scale = 2

    # Draw on separate background if specified
    if bgFile:
        imContours = drawColonies(im, colContours, cv.imread(bgFile), labelType='id', scale=scale)
    else:
        imContours = drawColonies(im, colContours, labelType='id', scale=scale)
    cv.imshow('GV Colony IDs (Press any key to exit)', imContours)

    filename = inFile[inFile.rfind('/') + 1:inFile.rfind('.')]
    cv.waitKey(0)
    cv.destroyAllWindows()
    cv.imwrite(filename + '_colonies.png', imContours)
    cv.imwrite(filename + '_mask.png', segMask(im, colContours))


if __name__ == '__main__':
    if len(sys.argv) < 2 or len(sys.argv) > 3:
        print('usage: python colonyDetect.py <input image file> '
              '[optional background for visualization]')
        exit(1)
    inFile = sys.argv[1]
    if len(sys.argv) == 3:
        colonyAnalysis(inFile, sys.argv[2])
    else:
        colonyAnalysis(inFile)
