import os
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
results_folder = 'results'
subfolder = 'planes'
basefolder = os.path.join(SCRIPT_PATH, results_folder, subfolder)


def getFilename(name):
    return os.path.join(basefolder, name)


def resultsFolderImage(name):
    return lambda number: os.path.join(basefolder, name, str(number) + '.png')


def resultsFolderMat(name):
    return lambda number: os.path.join(basefolder, name, str(number) + '.mat')


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
rotatedBackmap = resultsFolderMat('rotatedBackmap')


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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Estimate equirectangular image depth.')
    parser.add_argument('-i', metavar='image', default='input.jpg')
    parser.add_argument('-nocrop', action='store_false')
    parser.add_argument('-nodepth', action='store_false')
    parser.add_argument('-noreproject', action='store_false')
    parser.add_argument('-noweighting', action='store_false')
    parser.add_argument('-nosphere', action='store_false')

    args = parser.parse_args()

    # config values
    image = args.i
    run_cropping = args.nocrop
    run_depth_prediction = args.nodepth
    run_reprojection = args.noreproject
    run_weighting = args.noweighting
    reconstruct_sphere = args.nosphere
    # reconstruct_from_raw_depth = False
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
        print("%s - Begin depth prediction..." % i)
        if (depthFayao(cropsFolder, depthFolder)):
            print("%s - Depth prediction OK." % i)
        else:
            print("%s - ERROR during depth prediction." % i)
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
            rotateBack(reprojected(i), alpha, beta, gamma,
                       writeToFile=rotatedBack(i), use_mat=True)
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
            depth_images, validmap_images, input_image)
        print('Weights calculated, interpolating...')
        weighted_depths = []
        for plane, weights in enumerate(plane_weights):
            left_weights = weights['left']
            right_weights = weights['right']
            top_L, bottom_L, left_L, right_L = weights['boundsLeft']
            top_R, bottom_R, left_R, right_R = weights['boundsRight']
            top_bound = top_L
            bottom_bound = bottom_L
            # Plot weights
            yL = np.array(range(left_weights.shape[0])) + top_bound
            pyplot.plot(yL, left_weights, label="%sL" % plane)
            pyplot.plot(yL, right_weights, label="%sR" % plane)
            pyplot.xlabel('index')
            pyplot.ylabel('weight')
            pyplot.grid(True)
            pyplot.legend(loc="best")

            weightsByCoord = {}
            left_bounds = np.zeros(len(left_weights))
            right_bounds = np.ones(len(right_weights)) * input_size[1]
            for [i, j] in weights['overlapLeft']:
                index = i - top_bound
                left_bounds[index] = max(left_bounds[index], j)
                weightsByCoord[(i, j)] = left_weights[index]
            for [i, j] in weights['overlapRight']:
                index = i - top_bound
                right_bounds[index] = min(right_bounds[index], j)
                weightsByCoord[(i, j)] = right_weights[index]
            raw = depth_images[plane]
            valid = validmap_images[plane]
            weighted_depth = np.zeros(raw.shape)

            def interpolate_weight(i,j, left_bound ,right_bound):
                left_weight = left_weights[i]
                right_weight = right_weights[i]
                column = float(j)
                if left_bound > right_bound:
                    right_bound += input_size[1]
                    if column < left_bound:
                        column += input_size[1]
                return linear_interpolate(column, left_bound, right_bound, left_weight, right_weight)

            max_depth = 0
            min_depth = 1000
            interp_region = 50
            for i in range(weighted_depth.shape[0]):
                for j in range(weighted_depth.shape[1]):
                    if valid.item(i, j) == 1:
                        depth_value = raw.item(i, j)
                        if (i, j) in weightsByCoord:
                            index = i - top_bound
                            left_bound = left_bounds[index]
                            right_bound = right_bounds[index]
                            diff = right_bound - left_bound
                            left_bound -= interp_region
                            right_bound += interp_region
                            if j > left_bound and j < right_bound:
                                weight = interpolate_weight(index, j, left_bound, right_bound)
                            else:
                                weight = weightsByCoord[(i, j)]
                            weighted_value = depth_value * weight
                        else:
                            if i < top_bound:
                                index = 0
                            elif i >= bottom_bound:
                                index = len(left_weights) - 1
                            else:
                                index = i - top_bound
                            left_bound = left_bounds[index] - interp_region
                            right_bound = right_bounds[index] + interp_region
                            weight = interpolate_weight(index, j, left_bound, right_bound)
                            weighted_value = depth_value * weight
                        weighted_depth.itemset(i, j, weighted_value)

            saveArray(weighted(plane), weighted_depth)
        pyplot.savefig(getFilename("weightsPlot.png"))
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
        # Reconstruct a colormapped depth
        # reconstructed = np.zeros((input_size[0], input_size[1], 3))
        # for k in range(len(planes)):
        #     for i in range(reconstructed.shape[0]):
        #         for j in range(reconstructed.shape[1]):
        #             pixel = planes[k].item(i, j)
        #             if pixel != 0:
        #                 reconstructed.itemset(i, j, (k % 2) * 2, pixel)
        # cv2.imwrite(getFilename('colormap.jpg'), reconstructed)

        difference = np.zeros((input_size[0], input_size[1]))
        average = np.zeros((input_size[0], input_size[1]))
        for i in range(difference.shape[0]):
            for j in range(difference.shape[1]):
                values = []
                for k in range(len(planes)):
                    pixel = planes.item(k, i, j)
                    if(pixel != 0):
                        values.append(pixel)
                if len(values) > 0:
                    average.itemset(i, j, sum(values) / len(values))
                    if len(values) == 2:
                        difference.itemset(i, j, abs(values[0] - values[1]))
        # Normalize difference
        imwriteNormalize(getFilename('difference.jpg'), difference)
        imwriteNormalize(getFilename('average.jpg'), average)

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

        saveArray(getFilename('blend.mat'), blend)
        imwriteNormalize(getFilename('blend.jpg'), blend)
