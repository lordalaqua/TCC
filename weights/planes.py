import numpy as np


def planes_depth_weights(depths, color):
    """ Find coefficients for weighing depth images"""
    # iterate through edges to find the constraints
    num_planes = depths.shape[0]
    pairs = [(x, x + 1 if x + 1 < num_planes else 0) for x in range(num_planes)]
    weights = {}
    for planeL, planeR in pairs:
        # Initialize empty equation system
        system = []
        # One weight for each line and each plane of the pair
        # Find vertical boundaries
        bounds = (0, depths.shape[1])
        num_lines = bounds[1] - bounds[0]
        num_weights = num_lines * 2
        # Add equations for depths
        for i in range(depths.shape[1]):
            for j in range(depths.shape[2]):
                pixel_a = depths.item(planeL, i, j)
                pixel_b = depths.item(planeR, i, j)
                if (pixel_a != 0 and pixel_b != 0):
                    equation = np.zeros(num_weights)
                    equation.itemset(i, pixel_a)
                    equation.itemset(i + num_lines, -pixel_b)
                    system.append(equation)
                    # Add equations between weights
                    current = color.item(i, j)
                    if i > bounds[0]:
                        prev = color.item(i - 1, j)
                        # diff = current - prev
                        # Create equation w(i) - w(i-1) = diff
                        # Create equation w(i+num_lines) - w(i-1+num_lines) = diff
                    if i < bounds[1] - 1:
                        next = color.item(i + 1, j)
                        # diff = current - prev
                        # Create equation w(i) - w(i+1) = diff
                        # Create equation w(i+num_lines) - w(i+1+num_lines) = diff

        system = np.array(system)
        u, s, v = np.linalg.svd(system)
        weights[(planeL, planeR)] = v[-1]
    return s
