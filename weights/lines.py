import numpy as np
from numpy.linalg import svd

def find_weights(depths, validmap, color):
    """ Find coefficients for weighing depth images"""
    # iterate through edges to find the constraints
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
        for i, j in overlap:
            # Add equations for depths
            equation = np.zeros(num_weights)
            depth_a = depths.item(planeL, i, j)
            depth_b = depths.item(planeR, i, j)
            equation.itemset(i - overlap_top, depth_a)
            equation.itemset(i - overlap_top + num_lines, -depth_b)
            system.append(equation)
            # Add equations between weights
            if i < overlap_bottom:
                current = np.array([color.item(i, j, x) for x in range(3)])
                next = np.array([color.item(i + 1, j, x) for x in range(3)])
                difference = abs(sum(next - current))
                v = 5
                if difference < 10:
                    v = 50
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
    return weights_by_plane


