import matplotlib.image as mpimg
import numpy as np
import cv2
import glob
import pickle
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from skimage.feature import hog
from skimage.io import imread
from vd_functions import *
from sklearn.model_selection import train_test_split

# Read in cars and notcars
car_images = glob.glob('vehicles/*/*.png')
noncar_images = glob.glob('non-vehicles/*/*.png')

cars = []
for image in car_images:
    cars.append(image)
    
notcars = []
for image in noncar_images:
    notcars.append(image)
    
print("Number of car images:", len(cars), "  Number of non-car images:", len(notcars))

# Define training parameters
orient = 9  # HOG orientations
pix_per_cell = 8 # HOG pixels per cell
cell_per_block = 2 # HOG cells per block
color_space = 'YCrCb' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
hog_channel = "ALL" # Can be 0, 1, 2, or "ALL"
spatial_size = (16, 16) # Spatial binning dimensions
hist_bins = 16    # Number of histogram bins
spatial_feat = True # Spatial features on or off
hist_feat = True # Histogram features on or off
hog_feat = True # HOG features on or off

car_features = extract_features(cars, color_space=color_space, 
                        spatial_size=spatial_size, hist_bins=hist_bins, 
                        orient=orient, pix_per_cell=pix_per_cell, 
                        cell_per_block=cell_per_block, 
                        hog_channel=hog_channel, spatial_feat=spatial_feat, 
                        hist_feat=hist_feat, hog_feat=hog_feat)
notcar_features = extract_features(notcars, color_space=color_space, 
                        spatial_size=spatial_size, hist_bins=hist_bins, 
                        orient=orient, pix_per_cell=pix_per_cell, 
                        cell_per_block=cell_per_block, 
                        hog_channel=hog_channel, spatial_feat=spatial_feat, 
                        hist_feat=hist_feat, hog_feat=hog_feat)

X = np.vstack((car_features, notcar_features)).astype(np.float64)                        
# Fit a per-column scaler
X_scaler = StandardScaler().fit(X)
# Apply the scaler to X
scaled_X = X_scaler.transform(X)

# Define the labels vector
y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))

# Split up data into randomized training and test sets
rand_state = np.random.randint(0, 100)
X_train, X_test, y_train, y_test = train_test_split(scaled_X, y, test_size=0.2, random_state=rand_state)

print('Using:',orient,'orientations',pix_per_cell,
    'pixels per cell and', cell_per_block,'cells per block')
print('Feature vector length:', len(X_train[0]))
# Use a linear SVC 
svc = LinearSVC()
svc.fit(X_train, y_train)
# Check the score of the SVC
print('Test Accuracy of SVC = ', round(svc.score(X_test, y_test), 4))

# Save svc, X_scaler and training parameters in a pickle file
trained_pickle = {}
trained_pickle["svc"] = svc
trained_pickle["X_scaler"] = X_scaler
trained_pickle["orient"] = orient
trained_pickle["pix_per_cell"] = pix_per_cell
trained_pickle["cell_per_block"] = cell_per_block
trained_pickle["color_space"] = color_space
trained_pickle["hog_channel"] = hog_channel
trained_pickle["spatial_size"] = spatial_size
trained_pickle["hist_bins"] = hist_bins
trained_pickle["spatial_feat"] = spatial_feat
trained_pickle["hist_feat"] = hist_feat
trained_pickle["hog_feat"] = hog_feat
pickle.dump(trained_pickle, open( "train_result.p", "wb" ) )
