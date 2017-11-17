import os
import sys

import numpy as np
import cv2

from partition.rotateSphere import rotateSphere, rotateBack
from partition.perspectiveToEquirectangular import perspectiveToEquirectangular
from partition.equirectangularToPerspective import equirectangularToPerspective

from weights.cubeMap import cube_map_depth_weights

path = os.path.dirname(os.path.realpath(__file__))


def mapImage(function, image):
    mapped = np.zeros(image.shape)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            pixel = image.item((i, j))
            mapped.itemset((i, j), function(pixel))
    return mapped


def depthPrediction(image, output):
    return not os.system('python %s/depth/runNeuralNet.py %s %s  > nul 2>&1' % (path, image, output))


if __name__ == "__main__":
    if(len(sys.argv) < 2):
        image = 'input.jpg'
    else:
        image = sys.argv[1]

    def filename(
        name): return lambda number: os.path.join(path, 'results', 'partial', name, str(number) + '.png')
    rotated = filename('rotated')
    crop = filename('crop')
    depth = filename('depth')
    normalized = filename('normalized')
    reprojected = filename('reprojection')
    validmap = filename('validmap')
    output = filename('output')
    outputmap = filename('outputmap')

    fov_h = 90  # 60.0
    fov = (90, 90)
    crop_size = 640

    angles = [(0, 0), (0, 90), (0, 180), (0, 270), (90, 0), (-90, 0)]
    # phi = vertical angle, theta = horizontal angle
    depth_values = []
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

        print("%s - Begin depth prediction..." % i)
        if(depthPrediction(crop(i), depth(i))):
            print("%s - Depth prediction OK." % i)
        else:
            print("%s - ERROR during depth prediction." % i)
        depth_values.append(cv2.imread(depth(i), 0).astype(np.float32) / 255.0)

        s = cube_map_depth_weights(np.array(depth_values))
        print(s)
        new_depths = []
        for i in range(len(s) / 2):
            new_img = (depth_values[i] * s[2 * i]) + s[2 * i + 1]
            new_depths.append(new_img)
        new_depths = np.array(new_depths)
        maxValue = np.amax(new_depths)
        minValue = np.amin(new_depths)
        print("Max: %s, Min: %s" % (maxValue, minValue))
        for i in range(len(s) / 2):
            cv2.imwrite(normalized(i),
                        mapImage(
                            lambda p: (p - minValue) /
                (maxValue - minValue) * 255,
                new_depths[i]))

    for i, (phi, theta) in enumerate(angles):

        alpha, beta, gamma = np.radians([0, phi, -theta])

        if perspectiveToEquirectangular(normalized(i), rotated(i), fov_h, crop_size, crop_size, reprojected(i), validmap(i)):
            print('Reprojecting %s...' % i)
        else:
            print('ERROR projecting back to equirectangular.')

        rotateBack(reprojected(i), alpha, beta, gamma, writeToFile=output(i))
        rotateBack(validmap(i), alpha, beta, gamma, writeToFile=outputmap(i))
