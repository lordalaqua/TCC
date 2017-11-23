import numpy as np
""" Find weights for weighing depth images of a 360 image cube-map
        for "normalizing" the estimated depth

    The function expects a cube-map of a spherical image, where the following
    planes form the "images" array, where Pi is the i-th element in the array.
                  *------*
                  |  P4  |
                  |      |
                  *------*------*------*------*
                  |  P0  |  P1  |  P2  |  P3  |
                  |      |      |      |      |
                  *------*------*------*------*
                  |  P5  |
                  |      |
                  *------*
"""


def top(scan, edge_length):
    return scan, 0


def top_inv(scan, edge_length):
    return edge_length - 1 - scan, 0


def bottom(scan, edge_length):
    return scan, edge_length - 1


def bottom_inv(scan, edge_length):
    return edge_length - 1 - scan, edge_length - 1


def left(scan, edge_length):
    return 0, scan


def right(scan, edge_length):
    return edge_length - 1, scan


center_edges = [
    (0, right, 1, left),
    (1, right, 2, left),
    (2, right, 3, left),
    (3, right, 0, left)]

pole_edges = [
    (4, top, 2, top_inv),
    (4, left, 3, top),
    (4, bottom, 0, top),
    (4, right, 1, top_inv),
    (5, top, 0, bottom),
    (5, left, 3, bottom_inv),
    (5, bottom, 2, bottom_inv),
    (5, right, 1, bottom)]


def add_equations_to_system(system, edge, edge_length, images, num_weights):
    plane_a, direction_a, plane_b, direction_b = edge
    for scan in range(edge_length):
        pixel_a = images.item(plane_a, *direction_a(scan, edge_length))
        pixel_b = images.item(plane_b, *direction_b(scan, edge_length))
        equation = np.zeros(num_weights)
        equation.itemset(plane_a, pixel_a)
        equation.itemset(plane_b, -pixel_b)
        system.append(equation)
    return system


def single_weights(images, exclude_poles=True):
    num_images = (4 if exclude_poles else 6)
    num_weights = num_images

    assert(images.shape[0] == num_images)
    assert(images.shape[1] == images.shape[2])
    image_size = images.shape[1]

    # Non-pole edges are a special, simpler case.
    edges = center_edges
    # if poles are included, add edges
    if not exclude_poles:
        edges += pole_edges

    # Initialize empty equation system
    system = []

    # iterate through edges to find the constraints
    for (plane_a, direction_a, plane_b, direction_b) in edges:
        for scan in range(image_size):
            pixel_a = images.item(plane_a, *direction_a(scan, image_size))
            pixel_b = images.item(plane_b, *direction_b(scan, image_size))
            equation = np.zeros(num_weights)
            equation.itemset(plane_a, pixel_a)
            equation.itemset(plane_b, -pixel_b)
            system.append(equation)
    # Find the minimum weights to transform images
    system = np.array(system)
    u, s, v = np.linalg.svd(system)
    return v[-1]


def per_edge_weights(images, exclude_poles=True):
    num_images = (4 if exclude_poles else 6)

    assert(images.shape[0] == num_images)
    assert(images.shape[1] == images.shape[2])
    image_size = images.shape[1]

    # Non-pole edges are a special, simpler case.
    edges = center_edges
    # if poles are included, add edges
    if not exclude_poles:
        edges += pole_edges

    # Initialize empty equation system
    weights = {}
    # iterate through edges to find the constraints
    for (plane_a, direction_a, plane_b, direction_b) in edges:
        system = []
        for scan in range(image_size):
            pixel_a = images.item(plane_a, *direction_a(scan, image_size))
            pixel_b = images.item(plane_b, *direction_b(scan, image_size))
            equation = np.zeros(2)
            equation.itemset(0, pixel_a)
            equation.itemset(1, -pixel_b)
            system.append(equation)
        system = np.array(system)
        u, s, v = np.linalg.svd(system)
        weights[(plane_a, plane_b)] = v[-1]

    output = [(0, 0) for x in range(num_images)]
    for i in range(num_images):
        for (left_image, right_image), (left_weight, right_weight) in weights.iteritems():
            if i == left_image:
                output[i] = (output[i][0], left_weight)
            elif i == right_image:
                output[i] = (right_weight, output[i][1])
    return output
