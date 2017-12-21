from callMatlabScript import callMatlabScript


def equirectangularToPerspective(image, fov, crop_height, crop_width, output, use_mat=0):
    return callMatlabScript('getcrop', image, fov, crop_height,
                            crop_width, output, use_mat)
