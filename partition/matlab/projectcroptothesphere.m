function projectcroptothesphere(crop, equirectangular, fov, crop_height, crop_width, output, validmap, use_mat)

fov = pi * str2num(fov) / 180;
panorama = double(imread(equirectangular));
[panoH, panoW, panoCh] = size(panorama);

if use_mat ~= 0
   mat_struct = load(crop);
   perspective = double(mat_struct.data_obj);
else
    perspective = double(imread(crop));
end

if length(size(perspective)) < 3
  warped_image(:,:,1) = perspective;
  warped_image(:,:,2) = perspective;
  warped_image(:,:,3) = perspective;
else
  warped_image = perspective;
end

% warped_image = warped_image/255;
% warped_image = warped_image((crop_height-crop_width)/2+(1:crop_width),:,:);

[reconstructed valid_pixels] = imNormal2Sphere(warped_image, fov, panoW, panoH);
% Get only valid pixels
if length(size(perspective)) < 3
  reconstructed = reconstructed(:,:,1) .* double(valid_pixels);
else
  reconstructed = reconstructed .* double(repmat(valid_pixels,[1,1,3]));
end

if use_mat ~= 0
  data_obj = reconstructed;
  save('-mat7-binary', output, 'data_obj');
  data_obj = valid_pixels;
  save('-mat7-binary', validmap, 'data_obj');
else
  imwrite(valid_pixels, validmap);
  imwrite(reconstructed/255, output);
end

exit;
