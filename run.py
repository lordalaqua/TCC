import os
import errno
import sys

import numpy as np
from scipy.io import loadmat, savemat
import cv2

from partition.rotateSphere import rotateSphere, rotateBack
from partition.perspectiveToEquirectangular import perspectiveToEquirectangular
from partition.equirectangularToPerspective import equirectangularToPerspective
from matplotlib import pyplot

from weights.lines import find_weights
import argparse

# Config variables
SCRIPT_PATH = os.path.dirname(os.path.realpath(__file__))


def mapImage(function, image):
    mapped = np.zeros(image.shape)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            pixel = image.item((i, j))
            mapped.itemset((i, j), function(pixel, i, j))
    return mapped

def imwriteNormalize(filename, data):
    maxDiff = np.amax(data)
    minDiff = np.amin(data)
    normalized = mapImage(lambda p, i, j: (
        p - minDiff) / (maxDiff - minDiff) * 255, data)
    cv2.imwrite(filename, normalized)

def saveArray(filename, data):
    savemat(filename, {'data_obj': data})

def loadArray(filename):
    mat = loadmat(filename)
    return mat['data_obj']


def depthEigen(image, output):
    return not os.system('python %s/depth/runNeuralNet.py %s %s  > nul 2>&1' %
                         (SCRIPT_PATH, image, output))


def depthFayao(image, output):
    return not os.system('matlab -nodisplay -nosplash -nodesktop -r -wait "cd %s; demo_modified %s %s; exit;"' %
                         (os.path.join(SCRIPT_PATH, 'depth-fayao', 'demo'), image, output))


def linear_interpolate(x, x1, x0, y1, y0):
    return y0 + (x - x0) * ((y1 - y0) / (x1 - x0))

def create_dir_if_needed(directory):
    try:
        os.makedirs(directory)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Estimate equirectangular image depth.')
    parser.add_argument('-i', metavar='image', default='input.jpg')
    parser.add_argument('-f', metavar='results_folder', default='results')
    parser.add_argument('-o', metavar='output_folder', default='output')
    parser.add_argument('-nocrop', action='store_false')
    parser.add_argument('-nodepth', action='store_false')
    parser.add_argument('-noreproject', action='store_false')
    parser.add_argument('-noweighting', action='store_false')
    parser.add_argument('-nosphere', action='store_false')

    args = parser.parse_args()
    # Get config values from args
    image = args.i
    results_folder = args.f
    subfolder = args.o
    run_cropping = args.nocrop
    run_depth_prediction = args.nodepth
    run_reprojection = args.noreproject
    run_weighting = args.noweighting
    reconstruct_sphere = args.nosphere

    
    basefolder = os.path.join(SCRIPT_PATH, results_folder, subfolder)

    def getFilename(name):
        return os.path.join(basefolder, name)


    def resultsFolderImage(name):
        directory = os.path.join(basefolder, name)
        create_dir_if_needed(directory)
        return lambda number: os.path.join(directory, str(number) + '.png')


    def resultsFolderMat(name):
        directory = os.path.join(basefolder, name)
        create_dir_if_needed(directory)
        return lambda number: os.path.join(directory, str(number) + '.mat')


    def resultsFolderSubFile(folder, file):
        return lambda number: os.path.join(folder, str(number), file)


    rotated = resultsFolderImage('rotated')
    crop = resultsFolderImage('crop')
    cropsFolder = getFilename('crop')
    depthFolder = getFilename('depth')
    depth = resultsFolderSubFile(depthFolder, 'predict_depth.mat')
    weighted = resultsFolderMat('weighted')
    reprojected = resultsFolderMat('reprojection')
    validmap = resultsFolderMat('validmap')
    rotatedBack = resultsFolderMat('rotatedBack')
    rotatedBackImage = resultsFolderImage('rotatedBack')
    rotatedBackmap = resultsFolderMat('rotatedBackmap')

    
    calculate_weights = find_weights
    fov_h = 90
    crop_size = 640
    angles = [(0, x * 45) for x in range(8)]

    input_image = cv2.imread(image)
    input_size = input_image.shape

    # phi = vertical angle, theta = horizontal angle
    if (run_cropping):
        for i, (phi, theta) in enumerate(angles):
            print('Cropping at %s, %s' % (theta, phi))

            alpha, beta, gamma = np.radians([0, phi, -theta])

            rotateSphere(image, alpha, beta, gamma, writeToFile=rotated(i))
            print('Saving %s' % rotated(i))

            if equirectangularToPerspective(
                    rotated(i), fov_h, crop_size, crop_size, crop(i)):
                print('Saving %s' % crop(i))
            else:
                print('ERROR projecting perspective image.')
    else:
        print('Skipping crop...')

    if (run_depth_prediction):
        print("Begin depth prediction...")
        if (depthFayao(cropsFolder, depthFolder)):
            print(" Depth prediction OK." )
        else:
            print("ERROR during depth prediction.")
    else:
        print('Skipping depth prediction...')

    if(run_reprojection):
        for i, (phi, theta) in enumerate(angles):
            alpha, beta, gamma = np.radians([0, phi, -theta])
            if perspectiveToEquirectangular(
                    depth(i), rotated(i), fov_h, crop_size,
                    crop_size, reprojected(i), validmap(i), use_mat=1):
                print('Reprojecting %s...' % i)
            else:
                print('ERROR projecting back to equirectangular.')
            v = rotateBack(reprojected(i), alpha, beta, gamma,
                       writeToFile=rotatedBack(i), use_mat=True)
            # imwriteNormalize(rotatedBackImage(i), v)
            rotateBack(validmap(i), alpha, beta, gamma,
                       writeToFile=rotatedBackmap(i), use_mat=True)
    else:
        print('Skipping reprojection...')

    if (run_weighting):
        print('Begin weighting...')
        # Collect depth and validmap from reprojections
        depth_images = []
        validmap_images = []
        for i, (phi, theta) in enumerate(angles):
            depth_images.append(loadArray(rotatedBack(i)))
            validmap_images.append(loadArray(rotatedBackmap(i)))
        depth_images = np.array(depth_images)
        validmap_images = np.array(validmap_images)

        # Run solving algorithm
        plane_weights = calculate_weights(
            depth_images, validmap_images, input_image, weighted, getFilename("weights-plot.png"))
        
    else:
        print('Skipping weighting...')
    # Reconstruct sphere
    if (reconstruct_sphere):
        print('Reconstructing spherical image...')
        planes = []
        valid_pixels = []
        for i, (phi, theta) in enumerate(angles):
            planes.append(loadArray(weighted(i)))
            valid_pixels.append(loadArray(rotatedBackmap(i)))
        planes = np.array(planes)
        valid_pixels = np.array(valid_pixels)

        # # Save difference in overlaps and average
        # difference = np.zeros((input_size[0], input_size[1]))
        # average = np.zeros((input_size[0], input_size[1]))
        # for i in range(difference.shape[0]):
        #     for j in range(difference.shape[1]):
        #         values = []
        #         for k in range(len(planes)):
        #             pixel = planes.item(k, i, j)
        #             if(pixel != 0):
        #                 values.append(pixel)
        #         if len(values) > 0:
        #             average.itemset(i, j, sum(values) / len(values))
        #             if len(values) == 2:
        #                 difference.itemset(i, j, abs(values[0] - values[1]))
        # # Normalize difference
        # imwriteNormalize(getFilename('difference.jpg'), difference)
        # imwriteNormalize(getFilename('average.jpg'), average)

        blend = np.zeros((input_size[0], input_size[1]))
        pairs = [(x, x + 1 if x + 1 < len(angles) else 0) for x in range(len(angles))]
        for planeL, planeR in pairs:
            step = input_size[1] / len(planes)
            left = (planeL+4) % 8 * step
            right = (planeR+4) % 8 * step
            col_offset = 0
            if right < left:
                right += input_size[1]
                col_offset = input_size[1]

            for i in range(input_size[0]):
                for j in range(left,right):
                    validL = valid_pixels.item(planeL,i,j)
                    validR = valid_pixels.item(planeR,i,j)
                    pixel = 0
                    column = j
                    if validL and validR:
                        alpha = float(column - left)/float(right-left)
                        pixel = (1-alpha)*planes.item(planeL,i,j) + alpha * planes.item(planeR,i,j)
                    elif validL:
                        pixel = planes.item(planeL,i,j)
                    elif validR:
                        pixel = planes.item(planeR,i,j)
                    # set pixel
                    if pixel:
                        blend.itemset(i, j, pixel)

        saveArray(getFilename('depth.mat'), blend)
        imwriteNormalize(getFilename('depth.jpg'), blend)
