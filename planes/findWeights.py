import numpy as np


def planes_depth_weights(images, exclude_poles=False):
    """ Find coefficients for weighing depth images"""
    # Initialize empty equation system
    system = []
    # iterate through edges to find the constraints
    for plane_a in range(images.shape[0]):
        for plane_b in range(plane_a+1, images.shape[0]):
            for i in range(images.shape[1]):
                for j in range(images.shape[2]):
                    pixel_a = images.item(plane_a, i, j)
                    pixel_b = images.item(plane_b, i, j)
                    if(pixel_a != 0 and pixel_b != 0):
                        equation = np.zeros(num_coefficients)
                        equation.itemset(2*plane_a, pixel_a)
                        equation.itemset(2*plane_a+1, 1)
                        equation.itemset(2*plane_b, -pixel_b)
                        equation.itemset(2*plane_b+1, -1)
                        system.append(equation)

    # Find the minimum coefficients to transform images
    system = np.array(system)
    u,s,v = np.linalg.svd(system)
    return s;