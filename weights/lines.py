import math
import numpy as np
from numpy.linalg import svd
from matplotlib import pyplot
from scipy.io import savemat
import cv2

def linear_interpolate(x, x1, x0, y1, y0):
    return y0 + (x - x0) * ((y1 - y0) / (x1 - x0))

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
    normalized = mapImage(lambda p, i, j: (((
        p - minDiff) / (maxDiff - minDiff) * 255) if p != 0 else 0), data)
    # normalized = cv2.applyColorMap(normalized.astype(np.uint8),cv2.COLORMAP_JET)
    cv2.imwrite(filename, normalized)

def find_weights(depths, validmap, color, weighted_filename, plot_filename):
    """ Find coefficients for weighing depth images"""
    # iterate through edges to find the constraints
    input_size = color.shape
    num_planes = depths.shape[0]
    pairs = [(x, x + 1 if x + 1 < num_planes else 0)
             for x in range(num_planes)]
    weights = {}
    overlaps = {}
    overlap_bound = {}
    for planeL, planeR in pairs:
        # Find overlapping coordinates
        overlap = []
        overlap_top = validmap.shape[1]
        overlap_bottom = 0
        overlap_left = validmap.shape[2]
        overlap_right = 0
        for i in range(validmap.shape[1]):
            for j in range(validmap.shape[2]):
                pixel_a = validmap.item(planeL, i, j)
                pixel_b = validmap.item(planeR, i, j)
                if(pixel_a == 1 and pixel_b == 1):
                    overlap.append(np.array([i, j]))
                    overlap_top = min(i, overlap_top)
                    overlap_bottom = max(i, overlap_bottom)
                    overlap_left = min(j, overlap_left)
                    overlap_right = max(j, overlap_right)

        overlap = np.array(overlap)
        num_lines = overlap_bottom - overlap_top + 1
        num_weights = num_lines * 2
        # Initialize empty equation system
        system = []
        max_difference = 3*(255**2)
        threshold = max_difference / 10
        alpha = 1.0
        for i, j in overlap:
            # Add equations for depths
            equation = np.zeros(num_weights)
            depth_a = depths.item(planeL, i, j)
            depth_b = depths.item(planeR, i, j)
            equation.itemset(i - overlap_top, depth_a*alpha)
            equation.itemset(i - overlap_top + num_lines, -depth_b*alpha)
            system.append(equation)
            # Add equations between weights
            if i < overlap_bottom:
                current = np.array([color.item(i, j, x) for x in range(3)])
                next = np.array([color.item(i + 1, j, x) for x in range(3)])
                difference = sum((next - current)**2)               
                v = alpha * 50
                if difference < threshold:
                    v = alpha * 100
                equation = np.zeros(num_weights)
                equation.itemset(i - overlap_top, v)
                equation.itemset(i - overlap_top + 1, -v)
                system.append(equation)
                # Add equation for planeR weights
                equation = np.zeros(num_weights)
                equation.itemset(i - overlap_top + num_lines, v)
                equation.itemset(i - overlap_top + num_lines + 1, -v)
                system.append(equation)
        # Solve system
        print("Solving system for pair (%s,%s)..."% (planeL, planeR))
        system = np.array(system)
        u, s, v = svd(system, full_matrices=False)
        print("Solved!")
        solution = v[-1]
        weights[(planeL, planeR)] = (
            solution[:num_lines], solution[num_lines:])
        overlaps[(planeL, planeR)] = overlap
        overlap_bound[(planeL, planeR)] = (
            overlap_top, overlap_bottom, overlap_left, overlap_right)

    # Sort weights by plane
    weights_by_plane = [{} for x in range(len(weights))]
    for (planeL, planeR), (weightsL, weightsR) in weights.iteritems():
        weights_by_plane[planeL]['left'] = np.absolute(weightsL)
        weights_by_plane[planeL]['overlapRight'] = overlaps[(planeL, planeR)]
        weights_by_plane[planeL]['boundsRight'] = overlap_bound[(
            planeL, planeR)]
        weights_by_plane[planeR]['right'] = np.absolute(weightsR)
        weights_by_plane[planeR]['overlapLeft'] = overlaps[(planeL, planeR)]
        weights_by_plane[planeR]['boundsLeft'] = overlap_bound[(
            planeL, planeR)]

    print('Weights calculated, interpolating...')
    for plane, weights in enumerate(weights_by_plane):
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
        raw = depths[plane]
        valid = validmap[plane]
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

        savemat(weighted_filename(plane), {'data_obj': weighted_depth})
        # imwriteNormalize(weighted_filename(plane).replace('mat','png'), weighted_depth)
        # Save weights plot
        pyplot.xlabel('index')
        pyplot.ylabel('weight')
        pyplot.grid(True)
        pyplot.legend(loc="best")
        pyplot.savefig(plot_filename)


