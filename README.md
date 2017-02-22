##Vehicle detection and bounding boxes

The goals of this project are as following:
* Perform feature extraction (HOG, color etc.) on a labeled training set of images and train a classifier
* Implement a sliding-window technique and use the trained classifier to search for vehicles in test images
* Run image pipeline on a video stream, create a heat map of recurring detections frame by frame to reject outliers
* Estimate and draw a bounding box for each vehicle detected in the video

###Histogram of Oriented Gradients (HOG)

####1. Explain how (and identify where in your code) you extracted HOG features from the training images.

Udacity provided a set of `vehicle` and `non-vehicle` images as training dataset for this project.  Here are couple of examples of each of the `vehicle` and `non-vehicle` classes:
![](output_images/car_notcar.png)

The training data had 64x64x3 cutout images with 8792 car images and 8968 non-car images. Number of car and non-car images were roughly equal, so there was no need to do any furhter class balancing. The next step was to identify a set of features that would serve well for the classification task of identifying presence of a car in any 64x64x3 image. Histogram of Oriented Gradients (HOG) is one such feature, which has proven to be very useful in detecting objects of distinct shapes. So, HOG was one obvious choice for the set of features. The parameters of HOG are `orientations`, `pixels per cells`, `cells per block`, and the `channels` of the `color space` on which to calculate HOG. 


