import math
import numpy as np
# from scipy.sparse.linalg import svds as svd
from numpy.linalg import svd
from scipy.io import savemat
from scipy import interpolate 
import cv2


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
    

def find_weights(depths_original, validmap_original, color_original, weighted_filename, plot_filename):
    """ Find coefficients for weighing depth images"""
    # iterate through edges to find the constraints
    scale = 8
    original_size = (color_original.shape[1], color_original.shape[0])
    new_size = (color_original.shape[1]/scale, color_original.shape[0]/scale)
    depths = []
    validmap = []
    for i in range(len(depths_original)):
        depths.append(cv2.resize(depths_original[i], new_size, cv2.INTER_AREA))
        validmap.append(cv2.resize(validmap_original[i], new_size, cv2.INTER_AREA))
    depths = np.array(depths)
    validmap = np.array(validmap)
    color = cv2.resize(color_original,new_size, cv2.INTER_AREA)
    input_size = color.shape
    num_planes = depths.shape[0]
    pairs = [(x, x + 1 if x + 1 < num_planes else 0)
             for x in range(num_planes)]
    weights_by_pair = {}
    overlaps = {}
    overlap_bound = {}
    for planeL, planeR in pairs:
        # Find overlapping coordinates
        overlap = []
        overlap_index = {}
        overlap_top = validmap.shape[1]
        overlap_bottom = 0
        overlap_left = validmap.shape[2]
        overlap_right = 0
        for i in range(validmap.shape[1]):
            for j in range(validmap.shape[2]):
                pixel_a = validmap_original.item(planeL, i*8, j*8)
                pixel_b = validmap_original.item(planeR, i*8, j*8)
                if(pixel_a == 1 and pixel_b == 1):
                    overlap.append((i,j))
                    overlap_index[(i, j)] = len(overlap)-1
                    overlap_top = min(i, overlap_top)
                    overlap_bottom = max(i, overlap_bottom)
                    overlap_left = min(j, overlap_left)
                    overlap_right = max(j, overlap_right)
        len_overlap = len(overlap)
        num_weights = len(overlap) * 2

        # Initialize empty equation system
        system = []
        for index, (i, j) in enumerate(overlap):
            # Add equations for depths
            alpha = 10.0
            equation = np.zeros(num_weights)
            depth_a = depths.item(planeL, i, j)
            depth_b = depths.item(planeR, i, j)
            equation.itemset(index, depth_a*alpha)
            equation.itemset(index + len_overlap, -depth_b*alpha)
            system.append(equation)
            # add equations for neighbors
            neighbors = [(i+1,j), (i,j+1), (i,j-1), (i-1,j), (i+1,j-1), (i+1,j+1), (i-1,j-1), (i-1,j+1)]
            current_color = np.array([color.item(i, j, x) for x in range(3)])
            max_difference = 3*(255**2)
            threshold = max_difference / 10
            num_neighbors = 0
            for neighbor in neighbors:
                if neighbor in overlap_index:
                    num_neighbors += 1
            
            for neighbor in neighbors:
                if neighbor in overlap_index:                    
                    (m,n) = neighbor
                    neighbor_index = overlap_index[neighbor]
                    neighbor_color = np.array([color.item(m, n, x) for x in range(3)])
                    difference = sum((neighbor_color - current_color)**2)
                    v = alpha/4
                    if difference < threshold:
                        v = alpha
                    #     v =  alpha*2 #(max_difference-difference) / max_difference * alpha * 100 #*(10-num_neighbors) * 10
                    equation = np.zeros(num_weights)
                    equation.itemset(index, v)
                    equation.itemset(neighbor_index, -v)
                    system.append(equation)
                    equation = np.zeros(num_weights)
                    equation.itemset(index + len_overlap, v)
                    equation.itemset(neighbor_index + len_overlap, -v)
                    system.append(equation)
        
        # Solve system
        
        print("Solving system for pair (%s,%s)..."% (planeL, planeR))
        system = np.array(system)
        u, s, v = svd(system)        
        print("Solved!")
        solution = v[-1]
        weights_by_coordL = {}
        weights_by_coordR = {}
        for index, (i, j) in enumerate(overlap):
            weights_by_coordL[(i,j)] = solution[index]
            weights_by_coordR[(i,j)] = solution[index + len_overlap]
        weights_by_pair[(planeL, planeR)] = (weights_by_coordL, weights_by_coordR)
        overlaps[(planeL, planeR)] = overlap
        overlap_bound[(planeL, planeR)] = (
            overlap_top, overlap_bottom, overlap_left, overlap_right)
    # Sort weights by plane
    weight_images = np.zeros(depths.shape)
    for (planeL, planeR), (weightsL, weightsR) in weights_by_pair.iteritems():
        for plane, weights in [(planeL,weightsL),(planeR, weightsR)]:
            for (i,j), weight in weights.iteritems():
                weight_images.itemset(plane,i,j, abs(weight))

    for plane in range(len(weight_images)):
        weight_image = weight_images[plane]
        if plane == 4:
            num_cols = weight_image.shape[1]
            step = num_cols/8
            new_image = np.zeros(weight_image.shape)
            new_image[:,3*step:4*step] = weight_image[:,7*step:]
            new_image[:,4*step:5*step] = weight_image[:,:step]
            weight_image = new_image        
        rows = []
        cols = []
        interp_values = []
        weights = {}
        for i in range(weight_images.shape[1]):
            for j in range(weight_images.shape[2]):
                weight = weight_image.item(i, j)
                if(weight != 0):
                    rows.append(i)
                    cols.append(j)
                    interp_values.append(weight)
                    weights[(i,j)] = weight
        print('Interpolating %s...' % plane)
        grid_x, grid_y = np.meshgrid(range(weight_images.shape[1]),range(weight_images.shape[2]))        
        # interpolated = interpolate.griddata(np.array([rows, cols]).T, np.array(interp_values).T, (grid_x, grid_y) , method='linear', fill_value=.0)
        interp_func = interpolate.Rbf(np.array(rows),np.array(cols),np.array(interp_values), function='linear')
        interpolated = interp_func(grid_x, grid_y).T
        if plane == 4:
            num_cols = weight_image.shape[1]
            step = num_cols/8
            new_image = np.zeros(interpolated.shape)            
            new_image[:,7*step:] = interpolated[:,3*step:4*step]
            new_image[:,:step] = interpolated[:,4*step:5*step]
            interpolated = new_image
        weight_images[plane] = interpolated
        
    # Resize weight grid to original image size

    resized_weights = []
    for i in range(len(weight_images)):
        resized = cv2.resize(weight_images[i],original_size)
        clipped = cv2.bitwise_and(resized,resized, mask=validmap_original[i])
        resized_weights.append(clipped)
        imwriteNormalize('%sweight.jpg'%i, resized_weights[i])
    resized_weights = np.array(resized_weights)

    # resized_weights = np.zeros(depths_original.shape)
    # for plane in range(resized_weights.shape[0]):
    #     for i in range(resized_weights.shape[1]):
    #         for j in range(resized_weights.shape[2]):
    #             if validmap_original.item(plane,i,j) == 1:
    #                 resized_weights.itemset(plane,i,j,weight_images.item(plane,i/scale,j/scale))
    #     imwriteNormalize('%sweight.jpg'%plane, resized_weights[plane])


    print('Weights calculated, interpolating...')
    for plane in range(len(resized_weights)):
        # imwriteNormalize('%sinterp.jpg'%plane, interpolated)
        raw = depths_original[plane]
        valid = validmap_original[plane]
        weighted_depth = np.zeros(raw.shape)

        for i in range(weighted_depth.shape[0]):
            for j in range(weighted_depth.shape[1]):
                if valid.item(i, j) == 1:
                    weight = resized_weights.item(plane,i,j)
                    if weight > 0:
                        depth_value = raw.item(i, j)
                        weighted_value = depth_value * weight
                        weighted_depth.itemset(i, j, weighted_value)
        savemat(weighted_filename(plane), {'data_obj': weighted_depth})
        imwriteNormalize(weighted_filename(plane).replace('mat','png'), weighted_depth)
