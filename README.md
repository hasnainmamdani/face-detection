# face-detection

### Overview
This is a school project in which we (myself and group mate) implement the sliding window model for face detection. This technique is conceptually simple: independently classify all image patches as being object or non-object. Sliding window classification is the dominant paradigm in object detection and for one object category in particular -- faces -- it is one of the most noticeable successes of computer vision. For example, modern cameras and photo organization tools have prominent face detection capabilities. For this project we implemented the simpler (but still very effective!) sliding window detector of Dalal and Triggs 2005. Dalal-Triggs focuses on representation more than learning and introduces the SIFT-like Histogram of Gradients (HoG) representation (pictured below). We do not implement HoG but the rest of the detection pipeline -- handling heterogeneous training and testing data, training a linear classifier (a HoG template), and using your classifier to classify millions of sliding windows at multiple scales. Fortunately, linear classifiers are compact, fast to train, and fast to execute. A linear SVM can also be trained on large amounts of data, including mined hard negatives.
![](https://github.com/hasnainmamdani/face-detection/blob/master/html/hog_vis.png)

### Code Structure
The skeleton code was provided to us by the professor. We kept the file names unchanged and implemented the missing parts. Following is an outline of the stencil code:

*proj4.m*. The top level script for training and testing your object detector. It predicts random faces in the test images. It calls the following functions, many of which are simply placeholders in the starter code.

get_positive_features.m (we coded this). Load cropped positive trained examples (faces) and convert them to HoG features with a call to vl_hog.

*get_random_negative_features.m* (we coded this). Sample random negative examples from scenes which contain no faces and convert them to HoG features.

*classifier training* (we coded this in proj4.m). Train a linear classifier from the positive and negative examples with a call to vl_trainsvm.

*run_detector.m* (we coded this). Run the classifier on the test set. For each image, run the classifier at multiple scales and then call non_max_supr_bbox to remove duplicate detections.

*evaluate_detections.m*. Compute ROC curve, precision-recall curve, and average precision. You're not allowed to change this function.

*visualize_detections_by_image.m*. Visualize detections in each image. You can use visualize_detections_by_image_no_gt.m for test cases which have no ground truth annotations (e.g. the class photos).

### Data
The choice of training data is critical for this task. While an object detection system would typically be trained and tested on a single database (as in the Pascal VOC challenge), face detection papers have traditionally trained on heterogeneous, even proprietary, datasets. As with most of the literature, we will use three databases: (1) positive training crops, (2) non-face scenes to mine for negative training data, and (3) test scenes with ground truth face locations. The datasets were also kindly provided to us by the professor.

We used a positive training database of 6,713 cropped 36x36 faces from the Caltech Web Faces project. We arrived at this subset by filtering away faces which were not high enough resolution, upright, or front facing.

Non-face scenes, the second source of your training data, are easy to collect. We used a small database of such scenes from the SUN scene database. You can add more non-face training scenes, although you are unlikely to need more negative training data unless you are doing hard negative mining for extra credit.

The most common benchmark for face detection is the CMU+MIT test set. This test set contains 130 images with 511 faces. The test set is challenging because the images are highly compressed and quantized. Some of the faces are illustrated faces, not human faces. For this project, we have converted the test set's ground truth landmark points in to Pascal VOC style bounding boxes. We have inflated these bounding boxes to cover most of the head, as the provided training data does. For this reason, you are arguably training a "head detector" not a "face detector" for this project.

### Implementation Details and Results
Please see the html we have prepared: http://htmlpreview.github.io/?https://github.com/hasnainmamdani/face-detection/blob/master/html/index.html
