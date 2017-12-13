import os
import sys

import numpy as np
from scipy import interpolate
import cv2

from partition.rotateSphere import rotateSphere, rotateBack
from partition.perspectiveToEquirectangular import perspectiveToEquirectangular
from partition.equirectangularToPerspective import equirectangularToPerspective

from weights.planes import find_weights

path = os.path.dirname(os.path.realpath(__file__))


def mapImage(function, image):
    mapped = np.zeros(image.shape)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            pixel = image.item((i, j))
            mapped.itemset((i, j), function(pixel, i, j))
    return mapped


def depthEigen(image, output):
    return not os.system('python %s/depth/runNeuralNet.py %s %s  > nul 2>&1' %
                         (path, image, output))


def depthFayao(image, output):
    return not os.system('matlab -nodisplay -nosplash -nodesktop -r -wait "cd %s; demo_modified %s %s; exit;"' %
                         (os.path.join(path, 'depth-fayao', 'demo'), image, output))


if __name__ == "__main__":
    if (len(sys.argv) < 2):
        image = 'input.jpg'
    else:
        image = sys.argv[1]
    input_image = cv2.imread(image)
    input_size = input_image.shape

    # config values
    run_cropping = False
    run_depth_prediction = False
    run_reprojection = False
    run_weighting = False
    reconstruct_sphere = True
    # reconstruct_from_raw_depth = False
    calculate_weights = find_weights
    fov_h = 90
    crop_size = 640
    angles = [(0, x * 45) for x in range(8)]
    results_folder = 'results'
    subfolder = 'planes'
    basefolder = os.path.join(path, results_folder, subfolder)

    # Define files and folder names for storing result images
    def files(name):
        return os.path.join(basefolder, name)

    def folders(name):
        return lambda number: os.path.join(basefolder, name, str(number) + '.png')

    rotated = folders('rotated')
    crop = folders('crop')
    cropsFolder = files('crop')
    depthFolder = files('depth')

    def depth(number): return os.path.join(
        depthFolder, str(number), 'predict_depth_gray.png')

    weighted = folders('weighted')
    reprojected = folders('reprojection')
    validmap = folders('validmap')
    rotatedBack = folders('rotatedBack')
    rotatedBackmap = folders('rotatedBackmap')
    reconstruction = folders('reconstruction')

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
            yL = np.array(range(weights['left'].shape[0])) + weights['bound']
            xL = np.ones(yL.shape) * left_edge
            yR = np.array(range(weights['right'].shape[0])) + weights['bound']
            xR = np.ones(yR.shape) * right_edge
            y = np.append(yL, yR)
            x = np.append(xL, xR)
            z = np.append(weights['left'], weights['right'])
            interpolated = interpolate.interp2d(x, y, z)
            raw = cv2.imread(reprojected(plane), 0)
            valid = cv2.imread(validmap(plane), 0)
            weighted_depth = np.zeros(raw.shape)
            for i in range(raw.shape[0]):
                for j in range(raw.shape[1]):
                    if(valid.item(i, j) == 255):
                        weighted_depth.itemset(i, j, raw.item(
                            i, j) * interpolated(j, i))
            weighted_depths.append(weighted_depth)
        print('Interpolation complete, normalizing...')
        weighted_depths = np.array(weighted_depths)
        maxValue = np.amax(weighted_depths)
        minValue = np.amin(weighted_depths)

        def normalize(p):
            if p != 0:
                return (p - minValue) / (maxValue - minValue) * 255
            else:
                return 0
        for i in range(len(plane_weights)):
            weighted_image = mapImage(
                lambda p, i, j: normalize(p), weighted_depths[i])
            cv2.imwrite(weighted(i), weighted_image)
    # Reconstruct sphere
    if (reconstruct_sphere):
        reconstructed = np.zeros((input_size[0], input_size[1]))
        planes = []
        for i, (phi, theta) in enumerate(angles):
            alpha, beta, gamma = np.radians([0, phi, -theta])
            rotateBack(
                weighted(i), alpha, beta, gamma, writeToFile=reconstruction(i))
            planes.append(cv2.imread(reconstruction(i), 0))

        def averageImages(p, i, j):
            values = []
            for k in range(len(planes)):
                v = planes[k].item(i, j)
                if(v != 0):
                    values.append(v)
            if(len(values) > 0):
                return values[-1]  # sum(values) / float(len(values))
            else:
                return 0

        reconstructed = mapImage(averageImages, reconstructed)
        cv2.imwrite(
            files('reconstructed_sphere.jpg'), reconstructed)
