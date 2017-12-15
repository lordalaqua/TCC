import os
import sys

import numpy as np
from scipy import interpolate
import cv2

from partition.rotateSphere import rotateSphere, rotateBack
from partition.perspectiveToEquirectangular import perspectiveToEquirectangular
from partition.equirectangularToPerspective import equirectangularToPerspective
from matplotlib import pyplot

from weights.planes import find_weights
import argparse

# Config variables
SCRIPT_PATH = os.path.dirname(os.path.realpath(__file__))
results_folder = 'results'
subfolder = 'planes'
basefolder = os.path.join(SCRIPT_PATH, results_folder, subfolder)


def getFilename(name):
    return os.path.join(basefolder, name)


def resultsFolder(name):
    return lambda number: os.path.join(basefolder, name, str(number) + '.png')


def resultsFolderSubFile(folder, file):
    return lambda number: os.path.join(folder, str(number), file)


rotated = resultsFolder('rotated')
crop = resultsFolder('crop')
cropsFolder = getFilename('crop')
depthFolder = getFilename('depth')
depth = resultsFolderSubFile(depthFolder, 'predict_depth_gray.png')
weighted = resultsFolder('weighted')
reprojected = resultsFolder('reprojection')
validmap = resultsFolder('validmap')
rotatedBack = resultsFolder('rotatedBack')
rotatedBackmap = resultsFolder('rotatedBackmap')
reconstruction = resultsFolder('reconstruction')


def mapImage(function, image):
    mapped = np.zeros(image.shape)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            pixel = image.item((i, j))
            mapped.itemset((i, j), function(pixel, i, j))
    return mapped


def depthEigen(image, output):
    return not os.system('python %s/depth/runNeuralNet.py %s %s  > nul 2>&1' %
                         (SCRIPT_PATH, image, output))


def depthFayao(image, output):
    return not os.system('matlab -nodisplay -nosplash -nodesktop -r -wait "cd %s; demo_modified %s %s; exit;"' %
                         (os.path.join(SCRIPT_PATH, 'depth-fayao', 'demo'), image, output))


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

    if (run_depth_prediction):
        print("%s - Begin depth prediction..." % i)
        if (depthFayao(cropsFolder, depthFolder)):
            print("%s - Depth prediction OK." % i)
        else:
            print("%s - ERROR during depth prediction." % i)

    if(run_reprojection):
        for i, (phi, theta) in enumerate(angles):
            alpha, beta, gamma = np.radians([0, phi, -theta])
            if perspectiveToEquirectangular(
                    reconstruct_in(i), rotated(i), fov_h, crop_size,
                    crop_size, reprojected(i), validmap(i)):
                print('Reprojecting %s...' % i)
            else:
                print('ERROR projecting back to equirectangular.')
            rotateBack(
                reprojected(i), alpha, beta, gamma, writeToFile=rotatedBack(i))
            rotateBack(
                validmap(i), alpha, beta, gamma, writeToFile=rotatedBackmap(i))

    if (run_weighting):
        print('Begin weighting...')
        # Collect depth and validmap from reprojections
        depth_images = []
        validmap_images = []
        for i, (phi, theta) in enumerate(angles):
            depth_images.append(cv2.imread(
                rotatedBack(i), 0).astype(np.float32))
            validmap_images.append(cv2.imread(rotatedBackmap(i), 0))
        depth_images = np.array(depth_images)
        validmap_images = np.array(validmap_images)

        # Run solving algorithm
        plane_weights = calculate_weights(
            depth_images, validmap_images, input_image)
        print('Weights calculated, interpolating...')
        # Interpolate centered depths to avoid interpolation over image edge
        left_edge = input_size[1] / 8 * 3.0
        right_edge = input_size[1] / 8 * 5.0
        weighted_depths = []
        for plane, weights in enumerate(plane_weights):

            # xL = np.ones(yL.shape) * left_edge
            # yR = np.array(range(weights['right'].shape[0])) + weights['bound']
            # xR = np.ones(yR.shape) * right_edge
            # y = np.append(yL, yR)
            # x = np.append(xL, xR)
            # z = np.append(weights['left'], weights['right'])
            # interpolated = interpolate.interp2d(x, y, z)
            left_weights = weights['left']
            right_weights = weights['right']
            min_bound, max_bound = weights['bound']
            # Plot weights
            yL = np.array(range(left_weights.shape[0])) + min_bound
            pyplot.plot(yL, left_weights, label="%sL" % plane)
            pyplot.plot(yL, right_weights, label="%sR" % plane)
            pyplot.xlabel('index')
            pyplot.ylabel('weight')
            pyplot.grid(True)
            pyplot.legend(loc="best")
            raw = cv2.imread(rotatedBack(plane), 0)
            valid = cv2.imread(rotatedBackmap(plane), 0)
            weighted_depth = raw
            for [i, j] in weights['overlapLeft']:
                weighted_depth.itemset(i, j, raw.item(
                    i, j) * left_weights[i - min_bound])
            for [i, j] in weights['overlapRight']:
                weighted_depth.itemset(i, j, raw.item(
                    i, j) * right_weights[i - min_bound])
            # for i in range(weighted_depth.shape[0]):
            #     for j in range(weighted_depth.shape[1]):
            #         if [i, j] in weights['overlapLeft']:
            #             weighted_depth.itemset(i, j, weighted_depth.item(
            #                 i, j) * left_weights[i - min_bound])
            #         elif [i, j] in weights['overlapRight']:
            #             weighted_depth.itemset(i, j, weighted_depth.item(
            #                 i, j) * right_weights[i - min_bound])
            #         elif i > min_bound and i < max_bound:
            #             weighted_depth.itemset(i, j, weighted_depth.item(
            #                 i, j) * right_weights[i - min_bound])

            # def apply_weights(value, i, j):
            #     if [i, j] in :
            #         if(i - weights['bound'] > 241):
            #             print(i, j)
            #         return value *
            #     elif [i, j] in weights['overlapRight']:
            #         return value * weights['right'][i - weights['bound']]
            #     else:
            #         return value
                # if(valid.item(i, j) == 255):
                #     index = i - weights['bound']
                #     if(index > 0 and index < yL.shape[0]):
                #         if(j < input_size[1] / 2):
                #             weighted_depth.itemset(
                #                 i, j, weights['left'][index] * 1000)
                #         if(j > input_size[1] / 2):
                #             weighted_depth.itemset(
                #                 i, j, weights['right'][index] * 1000)
                # weighted_depth.itemset(i, j, raw.item(
                #     i, j) * interpolated(j, i))
            # weighted_depth = mapImage(apply_weights, raw)
            # for i in range(raw.shape[0]):
            #     for j in range(raw.shape[1]):

            weighted_depths.append(weighted_depth)
        pyplot.savefig(getFilename("weightsPlot.png"))
        print('Interpolation complete, normalizing...')
        weighted_depths = np.array(weighted_depths)
        maxValue = np.amax(weighted_depths)
        minValue = np.amin(weighted_depths)
        print('Min: %s, Max: %s' % (minValue, maxValue))

        def normalize(p):
            if p != 0:
                return (p - minValue) / (maxValue - minValue) * 255
            else:
                return 0
        for i in range(len(plane_weights)):
            # weighted_image = mapImage(
            #     lambda p, i, j: p, weighted_depths[i])
            cv2.imwrite(weighted(i), weighted_depths[i])
    # Reconstruct sphere
    if (reconstruct_sphere):
        print('Reconstructing spherical image...')
        planes = []
        for i, (phi, theta) in enumerate(angles):
            # alpha, beta, gamma = np.radians([0, phi, -theta])
            # rotateBack(
            #     weighted(i), alpha, beta, gamma, writeToFile=reconstruction(i))
            planes.append(cv2.imread(weighted(i), 0))

        # Reconstruct a colormapped depth
        reconstructed = np.zeros((input_size[0], input_size[1], 3))
        for k in range(len(planes)):
            for i in range(reconstructed.shape[0]):
                for j in range(reconstructed.shape[1]):
                    pixel = planes[k].item(i, j)
                    if pixel != 0:
                        reconstructed.itemset(i, j, k % 2 + 1, pixel)
        cv2.imwrite(getFilename('colormap.jpg'), reconstructed)

        difference = np.zeros((input_size[0], input_size[1]))
        average = np.zeros((input_size[0], input_size[1]))
        for i in range(difference.shape[0]):
            for j in range(difference.shape[1]):
                values = []
                for k in range(len(planes)):
                    pixel = planes[k].item(i, j)
                    if(pixel != 0):
                        values.append(pixel)
                if len(values) > 0:
                    average.itemset(i, j, sum(values) / len(values))
                    if len(values) == 2:
                        difference.itemset(i, j, abs(values[0] - values[1]))
        cv2.imwrite(getFilename('difference.jpg'), difference)
        cv2.imwrite(getFilename('average.jpg'), average)
