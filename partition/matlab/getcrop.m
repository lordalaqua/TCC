function getcrop(equirectangular, fov, crop_height, crop_width, output, use_mat)

fov = pi * str2num(fov) / 180;
if use_mat ~= 0
   mat_struct = double(load(equirectangular));
   panorama = mat_struct.data_obj;
else
    panorama = double(imread(equirectangular));
end

warped_image = imgLookAt(panorama, 0, 0, str2num(crop_height), fov);

% warped_image = warped_image((crop_height-crop_width)/2+(1:crop_width),:,:);
if use_mat ~= 0
    data_obj = warped_image
    save(output, 'data_obj')
else
    imwrite(warped_image, output);
end

exit;
