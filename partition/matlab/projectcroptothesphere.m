function projectcroptothesphere(crop, equirectangular, fov, crop_height, crop_width, output, validmap)

fov = pi * str2num(fov) / 180;

panorama = double(imread(equirectangular));
[panoH, panoW, panoCh] = size(panorama);

perspective = double(imread(crop));
if length(size(perspective)) < 3
  warped_image(:,:,1) = perspective;
  warped_image(:,:,2) = perspective;
  warped_image(:,:,3) = perspective;
else
  warped_image = perspective
end

% warped_image = warped_image/255;
% warped_image = warped_image((crop_height-crop_width)/2+(1:crop_width),:,:);

[reconstructed valid_pixels] = imNormal2Sphere(warped_image, fov, panoW, panoH);
% Get only valid pixels
reconstructed = reconstructed .* double(repmat(valid_pixels,[1,1,3]));

imwrite(valid_pixels, validmap);
imwrite(reconstructed/255, output);

exit;
