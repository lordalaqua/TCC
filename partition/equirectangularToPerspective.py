from callMatlabScript import callMatlabScript


def equirectangularToPerspective(image, fov, crop_height, crop_width, output):
    return callMatlabScript('getcrop', image, fov, crop_height,
                            crop_width, output)
