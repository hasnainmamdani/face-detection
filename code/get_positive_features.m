% Starter code prepared by James Hays for CS 143, Brown University
% This function should return all positive training examples (faces) from
% 36x36 images in 'train_path_pos'. Each face should be converted into a
% HoG template according to 'feature_params'. For improved performance, try
% mirroring or warping the positive training examples.

function features_pos = get_positive_features(train_path_pos, feature_params, enable_augment)
% 'train_path_pos' is a string. This directory contains 36x36 images of
%   faces
% 'feature_params' is a struct, with fields
%   feature_params.template_size (probably 36), the number of pixels
%      spanned by each train / test template and
%   feature_params.hog_cell_size (default 6), the number of pixels in each
%      HoG cell. template size should be evenly divisible by hog_cell_size.
%      Smaller HoG cell sizes tend to work better, but they make things
%      slower because the feature dimensionality increases and more
%      importantly the step size of the classifier decreases at test time.


% 'features_pos' is N by D matrix where N is the number of faces and D
% is the template dimensionality, which would be
%   (feature_params.template_size / feature_params.hog_cell_size)^2 * 31
% if you're using the default vl_hog parameters

% Useful functions:
% vl_hog, HOG = VL_HOG(IM, CELLSIZE)
%  http://www.vlfeat.org/matlab/vl_hog.html  (API)
%  http://www.vlfeat.org/overview/hog.html   (Tutorial)
% rgb2gray
fprintf(train_path_pos)
image_files = dir( fullfile( train_path_pos, '*.jpg') ); %Caltech Faces stored as .jpg

num_images = int16(length(image_files));

fprintf('\nget_positive_features use %d images\n',num_images)
% placeholder to be deleted
%feature_params = struct('template_size', 36, 'hog_cell_size', 6);
if enable_augment
    fprintf('with augmentation\n')
    features_pos = zeros(num_images*3, (feature_params.template_size / feature_params.hog_cell_size)^2 * 31);
else
    features_pos = zeros(num_images*2, (feature_params.template_size / feature_params.hog_cell_size)^2 * 31);
end

for i = 1:num_images
    img = imread(strcat(train_path_pos, '/', image_files(i).name));
    img = single(img)/255;
    if (size(img, 3) > 1)
        img = rgb2gray(img);
    end
    feat = vl_hog(img, feature_params.hog_cell_size);
    reshaped_feat = reshape(feat, 1, []);
    
    if enable_augment
        features_pos(3*i-1,:) = reshaped_feat;

        % mirroring the faces along the y axis
        feat_flip = vl_hog(flipdim(img, 2), feature_params.hog_cell_size);
        features_pos(3*i-2,:) = reshape(feat_flip, 1, []);
        
        img_rot = imrotate(img,-6,'bilinear','crop');
        img_rot = imcrop(img_rot, [3,3,30,30]);
        img_rot = imresize(img_rot, [36,36],'bilinear');
        feat_rot = vl_hog(img_rot, feature_params.hog_cell_size);
        features_pos(3*i,:) = reshape(feat_rot, 1,  []);
    else
        features_pos(2*i,:) = reshaped_feat;
        
        feat_flip = vl_hog(flipdim(img, 2), feature_params.hog_cell_size);
        features_pos(2*i-1,:) = reshape(feat_flip, 1, []);
    end
end

fprintf('get_positive_features done\n')


