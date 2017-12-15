import numpy as np


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
        overlap_min = validmap.shape[1]
        overlap_max = 0
        for i in range(validmap.shape[1]):
            for j in range(validmap.shape[2]):
                pixel_a = validmap.item(planeL, i, j)
                pixel_b = validmap.item(planeR, i, j)
                if(pixel_a == 255 and pixel_b == 255):
                    overlap.append(np.array([i, j]))
                    overlap_min = min(i, overlap_min)
                    overlap_max = max(i, overlap_max)

        overlap = np.array(overlap)
        num_lines = overlap_max - overlap_min + 1
        num_weights = num_lines * 2

        # Initialize empty equation system
        system = []
        for i, j in overlap:
            # Add equations for depths
            equation = np.zeros(num_weights)
            depth_a = depths.item(planeL, i, j)
            depth_b = depths.item(planeR, i, j)
            equation.itemset(i - overlap_min, depth_a)
            equation.itemset(i - overlap_min + num_lines, -depth_b)
            system.append(equation)
            # Add equations between weights
            if i < overlap_max:
                # current = np.array([color.item(i, j, x) for x in range(3)])
                # next = np.array([color.item(i + 1, j, x) for x in range(3)])
                # difference = abs(sum(next - current))
                v = 1000
                # Add equation for planeL weights
                equation = np.zeros(num_weights)
                equation.itemset(i - overlap_min, v)
                equation.itemset(i - overlap_min + 1, -v)
                system.append(equation)
                # print(equation)
                # raw_input("...")
                # Add equation for planeR weights
                equation = np.zeros(num_weights)
                equation.itemset(i - overlap_min + num_lines, v)
                equation.itemset(i - overlap_min + num_lines + 1, -v)
                system.append(equation)
        # Solve system
        system = np.array(system)
        u, s, v = np.linalg.svd(system, full_matrices=False)
        solution = v[-1] * 10
        weights[(planeL, planeR)] = (
            solution[:num_lines], solution[num_lines:])
        overlaps[(planeL, planeR)] = overlap
        overlap_bound[(planeL, planeR)] = (overlap_min, overlap_max)
    # Sort weights by plane
    weights_by_plane = [{} for x in range(len(weights))]
    for (planeL, planeR), (weightsL, weightsR) in weights.iteritems():
        weights_by_plane[planeL]['left'] = np.absolute(weightsL)
        weights_by_plane[planeL]['overlapLeft'] = overlaps[(planeL, planeR)]
        weights_by_plane[planeL]['bound'] = overlap_bound[(planeL, planeR)]
        weights_by_plane[planeR]['right'] = np.absolute(weightsR)
        weights_by_plane[planeR]['overlapRight'] = overlaps[(planeL, planeR)]
        weights_by_plane[planeR]['bound'] = overlap_bound[(planeL, planeR)]
    return weights_by_plane
