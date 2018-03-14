% Starter code prepared by James Hays for CS 143, Brown University
% This function should return negative training examples (non-faces) from
% any images in 'non_face_scn_path'. Images should be converted to
% grayscale, because the positive training data is only available in
% grayscale. For best performance, you should sample random negative
% examples at multiple scales.

function features_neg = get_random_negative_features(non_face_scn_path, feature_params, num_samples)
% 'non_face_scn_path' is a string. This directory contains many images
%   which have no faces in them.
% 'feature_params' is a struct, with fields
%   feature_params.template_size (probably 36), the number of pixels
%      spanned by each train / test template and
%   feature_params.hog_cell_size (default 6), the number of pixels in each
%      HoG cell. template size should be evenly divisible by hog_cell_size.
%      Smaller HoG cell sizes tend to work better, but they make things
%      slower because the feature dimensionality increases and more
%      importantly the step size of the classifier decreases at test time.
% 'num_samples' is the number of random negatives to be mined, it's not
%   important for the function to find exactly 'num_samples' non-face
%   features, e.g. you might try to sample some number from each image, but
%   some images might be too small to find enough.

% 'features_neg' is N by D matrix where N is the number of non-faces and D
% is the template dimensionality, which would be
%   (feature_params.template_size / feature_params.hog_cell_size)^2 * 31
% if you're using the default vl_hog parameters

% Useful functions:
% vl_hog, HOG = VL_HOG(IM, CELLSIZE)
%  http://www.vlfeat.org/matlab/vl_hog.html  (API)
%  http://www.vlfeat.org/overview/hog.html   (Tutorial)
% rgb2gray
fprintf(non_face_scn_path)
image_files = dir( fullfile( non_face_scn_path, '*.jpg' ));
num_images = length(image_files);
fprintf('\nget_random_negative_features use %d images\n',num_images)
% placeholder to be deleted
dims = (feature_params.template_size / feature_params.hog_cell_size)^2 * 31;
features_neg = zeros(num_samples, dims);

% Find number of samples per image
num_samples_per_image = ceil(num_samples / num_images);

% Size of each sample patch
patch_size = feature_params.template_size;

initial_num_samples = num_samples;
%[1 0.8 0.6 0.4]
scales = 1:-0.2:0.4;
count = 0 ;
for i = 1:num_images
    img = imread(strcat(non_face_scn_path, '/', image_files(i).name));
    if (size(img, 3) > 1)
        img = rgb2gray(img);
    end

    for scale_index = 1:length(scales)
        scale = scales(scale_index);
        scaled_img = imresize(img, scale);
        img_size = size(scaled_img);
        
        if img_size(1) < feature_params.template_size || img_size(2) < feature_params.template_size
            break
        end

        [y, x] = size(scaled_img);
	    for j = 1 : ceil(num_samples_per_image * scale)
	        small_img = img(randi(y - patch_size + 1) + (0 : patch_size - 1), randi(x - patch_size + 1) + (0 : patch_size - 1));
            feat = vl_hog(single(small_img), feature_params.hog_cell_size);
            reshaped_feat = reshape(feat, 1, []);
            count = count + 1;
            if count>=num_samples
                % resize features_neg
                num_samples = num_samples + initial_num_samples;
                temp = features_neg;
                features_neg = zeros(num_samples, dims);
                features_neg(1:size(temp, 1), :) = temp;
            end
	        features_neg(count, :) = reshaped_feat;
        end

    end
end
indices = randperm(count);
features_neg = features_neg(indices,:);
features_neg = features_neg(1:initial_num_samples, :);
fprintf('%d /%d neg patches are used\n', initial_num_samples, count );
fprintf('get_random_negative_features done\n\n');

