import numpy as np
import cv2
import pickle
import time
from vd_functions import *
from moviepy.editor import VideoFileClip
from scipy.ndimage.measurements import label

def get_heat(image):
    global svc, X_scaler, color_space, spatial_size, hist_bins, orient
    global pix_per_cell, cell_per_block, hog_channel, spatial_feat, hist_feat, hog_feat
    
    y_start_stop_1 = [400, 544]
    windows_1 = slide_window(image, x_start_stop=[None, None], y_start_stop=y_start_stop_1, 
                        xy_window=(96,96), xy_overlap=(0.5, 0.5))
    
    y_start_stop_2 = [528, 720]
    windows_2 = slide_window(image, x_start_stop=[None, None], y_start_stop=y_start_stop_2, 
                        xy_window=(128,128), xy_overlap=(0.5, 0.5))
    
    hot_windows_1 = search_windows(image, windows_1, svc, X_scaler, color_space=color_space, 
                        spatial_size=spatial_size, hist_bins=hist_bins, 
                        orient=orient, pix_per_cell=pix_per_cell, 
                        cell_per_block=cell_per_block, 
                        hog_channel=hog_channel, spatial_feat=spatial_feat, 
                        hist_feat=hist_feat, hog_feat=hog_feat) 

    hot_windows_2 = search_windows(image, windows_2, svc, X_scaler, color_space=color_space, 
                        spatial_size=spatial_size, hist_bins=hist_bins, 
                        orient=orient, pix_per_cell=pix_per_cell, 
                        cell_per_block=cell_per_block, 
                        hog_channel=hog_channel, spatial_feat=spatial_feat, 
                        hist_feat=hist_feat, hog_feat=hog_feat) 
    
    heat = np.zeros_like(image[:,:,0]).astype(np.float)
    heat = add_heat(heat, hot_windows_1)
    heat = add_heat(heat, hot_windows_2)
    
    return heat

def get_heat_sum(heat):
    global recent_heat_array
    
    recent_heat_array.append(heat)
    if len(recent_heat_array) > 10:
        recent_heat_array.pop(0)
        
    sum_h = np.zeros_like(heat).astype(np.float)
    for h in recent_heat_array:
        sum_h += h
    
    return sum_h

def process_image(image):
    '''Main image pipeline
    Args:
        image: Input image, or a frame from incoming video
    Returns:
        result: Output image, after vehicle detection bbox drawn back
    '''    
    image = image.astype(np.float32)/255
    heat = get_heat(image)
    heat_sum = get_heat_sum(heat)
    heat_sum[heat_sum <= 3] = 0
    labels = label(heat_sum)
    draw_img = draw_labeled_bboxes(255*np.copy(image), labels)
        
    return draw_img

# Global array to keep track of heatmaps of previous 10 frames
recent_heat_array = []

# Read the training result pickle file and load svc, X_scaler and training params
with open("train_result.p", mode='rb') as f:
    trained_pickle = pickle.load(f)

svc, X_scaler = trained_pickle["svc"], trained_pickle["X_scaler"]
orient = trained_pickle["orient"]
pix_per_cell = trained_pickle["pix_per_cell"]
cell_per_block = trained_pickle["cell_per_block"]
color_space = trained_pickle["color_space"]
hog_channel = trained_pickle["hog_channel"]
spatial_size = trained_pickle["spatial_size"]
hist_bins = trained_pickle["hist_bins"]
spatial_feat = trained_pickle["spatial_feat"]
hist_feat = trained_pickle["hist_feat"]
hog_feat = trained_pickle["hog_feat"]

project_output = 'project_output.mp4'
clip1 = VideoFileClip("project_video.mp4")
project_clip = clip1.fl_image(process_image)
project_clip.write_videofile(project_output, audio=False)
