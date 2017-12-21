from callMatlabScript import callMatlabScript


def perspectiveToEquirectangular(crop, equirectangular, fov, crop_height, crop_width, output, validmap, use_mat=0):
    return callMatlabScript('projectcroptothesphere', crop, equirectangular,
                            fov, crop_height, crop_width, output, validmap, use_mat)
