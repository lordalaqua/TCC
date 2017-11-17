import numpy as np


def cube_map_depth_weights(images, exclude_poles=False):
    """ Find coefficients for weighing depth images of a 360 image cube-map
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
    num_images = (4 if exclude_poles else 6)
    num_coefficients = 2 * num_images

    assert(images.shape[0] == num_images)
    assert(images.shape[1] == images.shape[2])
    image_size = images.shape[1]

    # Define scanning function for each edge direction
    def top(scan): return scan, 0

    def top_inv(scan): return image_size - 1 - scan, 0

    def bottom(scan): return scan, image_size - 1

    def bottom_inv(scan): return image_size - 1 - scan, image_size - 1

    def left(scan): return 0, scan

    def right(scan): return image_size - 1, scan

    # Non-pole edges are a special, simpler case.
    edges = [
        (0, right, 1, left),
        (1, right, 2, left),
        (2, right, 3, left),
        (3, right, 0, left)]

    # if poles are included, add edges
    if not exclude_poles:
        edges += [
            (4, top, 2, top_inv),
            (4, left, 3, top),
            (4, bottom, 0, top),
            (4, right, 1, top_inv),
            (5, top, 0, bottom),
            (5, left, 3, bottom_inv),
            (5, bottom, 2, bottom_inv),
            (5, right, 1, bottom)]

    # Initialize empty equation system
    system = []

    # iterate through edges to find the constraints
    for (plane_a, direction_a, plane_b, direction_b) in edges:
        for scan in range(image_size):
            pixel_a = images.item(plane_a, *direction_a(scan))
            pixel_b = images.item(plane_b, *direction_b(scan))
            equation = np.zeros(num_coefficients)
            equation.itemset(2 * plane_a, pixel_a)
            equation.itemset(2 * plane_a + 1, 1)
            equation.itemset(2 * plane_b, -pixel_b)
            equation.itemset(2 * plane_b + 1, -1)
            system.append(equation)
    # Find the minimum coefficients to transform images
    system = np.array(system)
    u, s, v = np.linalg.svd(system)
    return v[10]
