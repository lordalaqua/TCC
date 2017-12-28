function getcrop(equirectangular, fov, crop_height, crop_width, output)

fov = pi * str2num(fov) / 180;

panorama = double(imread(equirectangular));

warped_image = imgLookAt(panorama, 0, 0, str2num(crop_height), fov);
warped_image = warped_image/255;
% warped_image = warped_image((crop_height-crop_width)/2+(1:crop_width),:,:);
imwrite(warped_image, output);

exit;
