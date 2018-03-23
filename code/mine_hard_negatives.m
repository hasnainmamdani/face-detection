function features_hard_neg = mine_hard_negatives(non_face_scn_path, w, b, feature_params)

test_scenes = dir( fullfile( non_face_scn_path, '*.jpg' ));

%initialize these as empty and incrementally expand them.
features_hard_neg = zeros(0, (feature_params.template_size / feature_params.hog_cell_size)^2 * 31);

for i = 1:length(test_scenes)

    %fprintf('Detecting hard negatives in %s\n', test_scenes(i).name)
    img = imread( fullfile( non_face_scn_path, test_scenes(i).name ));
    img = single(img)/255;
    if(size(img,3) > 1)
        img = rgb2gray(img);
    end

    curr_confidences = zeros(0,1);
    curr_bboxes = zeros(0,4);
    curr_features_hard_neg = zeros(0, (feature_params.template_size / feature_params.hog_cell_size)^2 * 31);

    scale = 1;
    window_size = feature_params.template_size / feature_params.hog_cell_size;

    while scale >= 0.1
        hog = vl_hog(imresize(img, scale), feature_params.hog_cell_size);

        for j = 1: size(hog, 1) - window_size
            for k = 1: size(hog, 2) - window_size
                curr_features = hog(j: j + window_size - 1, k: k + window_size - 1, :);
                curr_features = reshape(curr_features, [1, window_size ^ 2 * 31]) * w + b;
                confidence = curr_features * w + b;

                if confidence > 0.5
                    x_min = k * feature_params.hog_cell_size;
                    y_min = j * feature_params.hog_cell_size;

                    curr_bboxes = [curr_bboxes; [x_min, y_min, x_min + feature_params.template_size, y_min + feature_params.template_size]./scale];
                    curr_confidences = [curr_confidences; confidence];
                    curr_features_hard_neg = [curr_features_hard_neg; curr_features];
                end
            end
        end

        scale = scale - 0.05;
    end

    features_hard_neg = [features_hard_neg; curr_features_hard_neg];

end