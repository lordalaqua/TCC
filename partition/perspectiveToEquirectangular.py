from callMatlabScript import callMatlabScript


def perspectiveToEquirectangular(crop, equirectangular, fov, crop_height, crop_width, output, validmap):
    return callMatlabScript('projectcroptothesphere', crop, equirectangular,
                            fov, crop_height, crop_width, output, validmap)
