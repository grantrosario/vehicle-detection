import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob
import random
import time
from collections import deque
from skimage.feature import hog
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from scipy.ndimage.measurements import label

#---------------------
# Constants
#---------------------
NBINS = 32
SPATIAL = 32
COLOR_SPACE = 'YCrCb'
VIS_HOG = False
FEATURE_VECTOR = True
ORIENT = 9
PIX_PER_CELL = 8
CELL_PER_BLOCK = 2
HOG_CHANNEL = 'ALL'
WIND_SIZE = 64
WIND_OVERLAP = 0.5
SPATIAL_FEAT = True
HIST_FEAT = True
HOG_FEAT = True
USE_SAMPLE = True
SAMPLE_SIZE = 100
SCALE = 1.5
CELLS_PER_STEP = 2

#---------------------

cars = glob.glob('car*/*.png')
notcars = glob.glob('non_car*/*.png')


def color_hist(img, nbins): #bins_range = (0,256)
    # Compute histograms of color channels separately
    channel1_hist = np.histogram(img[:,:,0], bins = nbins)
    channel2_hist = np.histogram(img[:,:,1], bins = nbins)
    channel3_hist = np.histogram(img[:,:,2], bins = nbins)
    # Concatenate histograms into a single feature vector
    hist_features = np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))
    # Return feature vector
    return hist_features

def bin_spatial(img, size):
    color1 = cv2.resize(img[:,:,0], (size, size)).ravel()
    color2 = cv2.resize(img[:,:,1], (size, size)).ravel()
    color3 = cv2.resize(img[:,:,2], (size, size)).ravel()
    return np.hstack((color1, color2, color3))

def get_hog_features(img, orient, pix_per_cell, cell_per_block, vis = False, feature_vec = True):
    # Call with two outputs if vis == True
    if vis == True:
        features, hog_image = hog(img, orientations = orient, pixels_per_cell = (pix_per_cell, pix_per_cell),
                              cells_per_block = (cell_per_block, cell_per_block), transform_sqrt = True,
                              visualise = True, feature_vector = False)
        return features, hog_image
    # Otherwise call with one output
    else:
        features = hog(img, orientations = orient, pixels_per_cell = (pix_per_cell, pix_per_cell),
                   cells_per_block = (cell_per_block, cell_per_block), transform_sqrt = True,
                   visualise = False, feature_vector = feature_vec)
        return features

def extract_features(imgs, color_space = COLOR_SPACE, orient = ORIENT, pix_per_cell = PIX_PER_CELL,
                     cell_per_block = CELL_PER_BLOCK, hog_channel = HOG_CHANNEL):
    # Create list to append feature vectors to
    all_features = []
    # Iterate through list of images
    for img in imgs:
        img_features = []
        # Read in each one by one
        image = mpimg.imread(img)
        # Apply color conversion if other than 'RGB'
        if color_space != 'RGB':
            if color_space == 'HSV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            elif color_space == 'LUV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2LUV)
            elif color_space == 'HLS':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
            elif color_space == 'YUV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
            elif color_space == 'YCrCb':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
        else: feature_image = np.copy(image)

        if SPATIAL_FEAT == True:
            spatial_features = bin_spatial(feature_image, SPATIAL)
            img_features.append(spatial_features)
        if HIST_FEAT == True:
            hist_features = color_hist(feature_image, NBINS)
            img_features.append(hist_features)
        # Call get_hog_features() with vis=False, feature_vec=True
        if HOG_FEAT == True:
            if hog_channel == 'ALL':
                hog_features = []
                for channel in range(feature_image.shape[2]):
                    hog_features.append(get_hog_features(feature_image[:,:,channel],
                                        orient, pix_per_cell, cell_per_block,
                                        vis=False, feature_vec=True))
                hog_features = np.ravel(hog_features)
            else:
                hog_features = get_hog_features(feature_image[:,:,hog_channel], orient,
                            pix_per_cell, cell_per_block, vis=False, feature_vec=True)
            img_features.append(hog_features)
        all_features.append(np.concatenate((img_features)))
    # Return list of feature vectors
    return all_features

def convert_color(img, conv='RGB2YCrCb'):
    if conv == 'RGB2YCrCb':
        return cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    if conv == 'BGR2YCrCb':
        return cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    if conv == 'RGB2LUV':
        return cv2.cvtColor(img, cv2.COLOR_RGB2LUV)

def find_cars(img, ystart, ystop, scale, svc, X_scaler, window = WIND_SIZE):
    draw_img = np.copy(img)
    heatmap = np.zeros_like(img[:,:,0]) # delete
    img = img.astype(np.float32)/255
    box_list = []

    img_tosearch = img[ystart:ystop,:,:]
    ctrans_tosearch = convert_color(img_tosearch, conv='RGB2YCrCb')
    if scale != 1:
        imshape = ctrans_tosearch.shape
        ctrans_tosearch = cv2.resize(ctrans_tosearch, (np.int(imshape[1]/SCALE), np.int(imshape[0]/SCALE)))

    ch1 = ctrans_tosearch[:,:,0]
    ch2 = ctrans_tosearch[:,:,1]
    ch3 = ctrans_tosearch[:,:,2]

    # Define blocks and steps as above
    nxblocks = (ch1.shape[1] // PIX_PER_CELL) - CELL_PER_BLOCK + 1
    nyblocks = (ch1.shape[0] // PIX_PER_CELL) - CELL_PER_BLOCK + 1
    nfeat_per_block = ORIENT*CELL_PER_BLOCK**2

    # 64 was the orginal sampling rate, with 8 cells and 8 pix per cell
    nblocks_per_window = (window // PIX_PER_CELL) - CELL_PER_BLOCK + 1
    cells_per_step = CELLS_PER_STEP  # Instead of overlap, define how many cells to step
    nxsteps = (nxblocks - nblocks_per_window) // CELLS_PER_STEP
    nysteps = (nyblocks - nblocks_per_window) // CELLS_PER_STEP

    # Compute individual channel HOG features for the entire image
    hog1 = get_hog_features(ch1, ORIENT, PIX_PER_CELL, CELL_PER_BLOCK, feature_vec=False)
    hog2 = get_hog_features(ch2, ORIENT, PIX_PER_CELL, CELL_PER_BLOCK, feature_vec=False)
    hog3 = get_hog_features(ch3, ORIENT, PIX_PER_CELL, CELL_PER_BLOCK, feature_vec=False)

    for xb in range(nxsteps):
        for yb in range(nysteps):
            features = []
            ypos = yb*CELLS_PER_STEP
            xpos = xb*CELLS_PER_STEP
            # Extract HOG for this patch
            hog_feat1 = hog1[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel()
            hog_feat2 = hog2[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel()
            hog_feat3 = hog3[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel()


            xleft = xpos*PIX_PER_CELL
            ytop = ypos*PIX_PER_CELL

            # Extract the image patch
            subimg = cv2.resize(ctrans_tosearch[ytop:ytop+window, xleft:xleft+window], (64,64))

            # Get features
            if SPATIAL_FEAT == True:
                spatial_features = bin_spatial(subimg, SPATIAL)
                features.append(spatial_features)
            if HIST_FEAT == True:
                hist_features = color_hist(subimg, NBINS)
                features.append(hist_features)
            if HOG_FEAT == True:
                hog_features = np.hstack((hog_feat1, hog_feat2, hog_feat3))
                features.append(hog_features)

            # Scale features and make a prediction
            test_features = X_scaler.transform(np.hstack((features)).reshape(1, -1))
            test_prediction = svc.predict(test_features)

            if test_prediction == 1:
                xbox_left = np.int(xleft*scale)
                ytop_draw = np.int(ytop*scale)
                win_draw = np.int(window*scale)
                box = (xbox_left, ytop_draw+ystart), (xbox_left+win_draw, ytop_draw+win_draw+ystart)
                box_list.append(box)
                cv2.rectangle(draw_img,(xbox_left, ytop_draw+ystart),(xbox_left+win_draw,ytop_draw+win_draw+ystart),(0,0,255),6)
                heatmap[ytop_draw+ystart:ytop_draw+win_draw+ystart, xbox_left:xbox_left+win_draw] += 1
    return heatmap

def add_heat(heatmap, bbox_list):
    # Iterate through list of bboxes
    for box in bbox_list:
        # Add += 1 for all pixels inside each bbox
        # Assuming each "box" takes the form ((x1, y1), (x2, y2))
        heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1

    # Return updated heatmap
    return heatmap# Iterate through list of bboxes

def apply_threshold(heatmap, threshold):
    # Zero out pixels below the threshold
    heatmap[heatmap <= threshold] = 0
    # Return thresholded map
    return heatmap

def draw_labeled_bboxes(img, labels):
    # Iterate through all detected cars
    for car_number in range(1, labels[1]+1):
        # Find pixels with each car_number label value
        nonzero = (labels[0] == car_number).nonzero()
        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Define a bounding box based on min/max x and y
        bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
        # Draw the box on the image
        cv2.rectangle(img, bbox[0], bbox[1], (0,0,255), 4)
    # Return the image
    return img
