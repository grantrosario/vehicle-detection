import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob
import random
import time
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
SPATIAL_FEAT = False
HIST_FEAT = False
HOG_FEAT = True
USE_SAMPLE = True
SAMPLE_SIZE = 500
SCALE = 1.5
CELLS_PER_STEP = 2

#---------------------

cars = glob.glob('car*/*.png')
notcars = glob.glob('non_car*/*.png')

def single_img_features(img, color_space=COLOR_SPACE, spatial_size=(SPATIAL, SPATIAL),
                        hist_bins=NBINS, orient=ORIENT,
                        pix_per_cell=PIX_PER_CELL, cell_per_block=CELL_PER_BLOCK, hog_channel=HOG_CHANNEL,
                        spatial_feat=SPATIAL_FEAT, hist_feat=HIST_FEAT, hog_feat=HOG_FEAT):
    #1) Define an empty list to receive features
    img_features = []
    #2) Apply color conversion if other than 'RGB'
    if color_space != 'RGB':
        if color_space == 'HSV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        elif color_space == 'LUV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2LUV)
        elif color_space == 'HLS':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
        elif color_space == 'YUV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
        elif color_space == 'YCrCb':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    else: feature_image = np.copy(img)
    #3) Compute spatial features if flag is set
    if SPATIAL_FEAT == True:
        spatial_features = bin_spatial(feature_image, SPATIAL)
        #4) Append features to list
        img_features.append(spatial_features)
    #5) Compute histogram features if flag is set
    if HIST_FEAT == True:
        hist_features = color_hist(feature_image, nbins=NBINS)
        #6) Append features to list
        img_features.append(hist_features)
    #7) Compute HOG features if flag is set
    if HOG_FEAT == True:
        if hog_channel == 'ALL':
            hog_features = []
            for channel in range(feature_image.shape[2]):
                hog_features.extend(get_hog_features(feature_image[:,:,channel],
                                    ORIENT, PIX_PER_CELL, CELL_PER_BLOCK,
                                    VIS_HOG, FEATURE_VECTOR))
        else:
            hog_features = get_hog_features(feature_image[:,:,hog_channel], ORIENT,
                        PIX_PER_CELL, CELL_PER_BLOCK, VIS_HOG, FEATURE_VECTOR)
        #8) Append features to list
        img_features.append(hog_features)

    #9) Return concatenated array of features
    return np.concatenate(img_features)


def color_hist(img, nbins, bins_range = (0,256)):
    channel1_hist = np.histogram(img[:,:,0], bins = nbins, range = bins_range)
    channel2_hist = np.histogram(img[:,:,1], bins = nbins, range = bins_range)
    channel3_hist = np.histogram(img[:,:,2], bins = nbins, range = bins_range)

    hist_features = np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))

    return hist_features

def bin_spatial(img, size):
    features = cv2.resize(img, (size, size)).ravel()
    return features

def get_hog_features(img, orient, pix_per_cell, cell_per_block, vis = False, feature_vec = True):
    if vis == True:
        features, hog_image = hog(img, orientations = orient, pixels_per_cell = (pix_per_cell, pix_per_cell),
                              cells_per_block = (cell_per_block, cell_per_block), transform_sqrt = True,
                              visualise = True, feature_vector = False)
        return features, hog_image
    else:
        features = hog(img, orientations = orient, pixels_per_cell = (pix_per_cell, pix_per_cell),
                   cells_per_block = (cell_per_block, cell_per_block), transform_sqrt = True,
                   visualise = False, feature_vector = feature_vec)
        return features

def extract_features(imgs, color_space = COLOR_SPACE, orient = ORIENT, pix_per_cell = PIX_PER_CELL, cell_per_block = CELL_PER_BLOCK, hog_channel = HOG_CHANNEL):
    all_features = []
    for img in imgs:
        img_features = []
        image = mpimg.imread(img)
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
            color_features = color_hist(feature_image, NBINS, (0, 256))
            img_features.append(color_features)
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
    return all_features

def slide_window(img, x_start_stop = [None, None], y_start_stop = [None, None], xy_window=(WIND_SIZE, WIND_SIZE), xy_overlap = (WIND_OVERLAP, WIND_OVERLAP)):
    if x_start_stop[0] == None:
        x_start_stop[0] = 0
    if x_start_stop[1] == None:
        x_start_stop[1] = img.shape[1]
    if y_start_stop[0] == None:
        y_start_stop[0] = 0
    if y_start_stop[1] == None:
        y_start_stop[1] = img.shape[0]

    # Region to be searched
    xspan = x_start_stop[1] - x_start_stop[0]
    yspan = y_start_stop[1] - y_start_stop[0]

    # Compute the number of pixels per step in x/y
    nx_pix_per_step = np.int(xy_window[0]*(1 - xy_overlap[0]))
    ny_pix_per_step = np.int(xy_window[1]*(1 - xy_overlap[1]))

    # Compute the number of windows in x/y
    nx_buffer = np.int(xy_window[0]*(xy_overlap[0]))
    ny_buffer = np.int(xy_window[1]*(xy_overlap[1]))
    nx_windows = np.int((xspan - nx_buffer)/nx_pix_per_step)
    ny_windows = np.int((yspan - ny_buffer)/ny_pix_per_step)

    window_list = []
    for ys in range(ny_windows):
        for xs in range(nx_windows):
            startx = xs*nx_pix_per_step + x_start_stop[0]
            endx = startx + xy_window[0]
            starty = ys*ny_pix_per_step + y_start_stop[0]
            endy = starty + xy_window[1]
            window_list.append(((startx, starty), (endx, endy)))
    return window_list

def draw_boxes(img, bboxes, color = (0, 0, 255), thick = 6):
    imcopy = np.copy(img)
    for bbox in bboxes:
        cv2.rectangle(imcopy, bbox[0], bbox[1], color, thick)
    return imcopy

def search_windows(img, windows, clf, scaler, color_space=COLOR_SPACE,
                    spatial_size=(SPATIAL, SPATIAL), hist_bins=NBINS,
                    hist_range=(0, 256), orient=ORIENT,
                    pix_per_cell=PIX_PER_CELL, cell_per_block=CELL_PER_BLOCK,
                    hog_channel=HOG_CHANNEL, spatial_feat=SPATIAL_FEAT,
                    hist_feat=HIST_FEAT, hog_feat=HOG_FEAT):

    #1) Create an empty list to receive positive detection windows
    on_windows = []
    #2) Iterate over all windows in the list
    for window in windows:
        #3) Extract the test window from original image
        test_img = cv2.resize(img[window[0][1]:window[1][1], window[0][0]:window[1][0]], (64, 64))
        #4) Extract features for that window using single_img_features()
        features = single_img_features(test_img, color_space=COLOR_SPACE,
                            spatial_size=SPATIAL, hist_bins=NBINS,
                            orient=ORIENT, pix_per_cell=PIX_PER_CELL,
                            cell_per_block=CELL_PER_BLOCK,
                            hog_channel=HOG_CHANNEL, spatial_feat=SPATIAL_FEAT,
                            hist_feat=HIST_FEAT, hog_feat=HOG_FEAT)
        #5) Scale extracted features to be fed to classifier
        test_features = scaler.transform(np.array(features).reshape(1, -1))
        #6) Predict using your classifier
        prediction = clf.predict(test_features)
        #7) If positive (prediction == 1) then save the window
        if prediction == 1:
            on_windows.append(window)
    #8) Return windows for positive detections
    return on_windows

def convert_color(img, conv='RGB2YCrCb'):
    if conv == 'RGB2YCrCb':
        return cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    if conv == 'BGR2YCrCb':
        return cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    if conv == 'RGB2LUV':
        return cv2.cvtColor(img, cv2.COLOR_RGB2LUV)

def find_cars(img, ystart, ystop, scale, svc, X_scaler):

    draw_img = np.copy(img)
    #img = img.astype(np.float32)/255

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
    window = WIND_SIZE
    nblocks_per_window = (WIND_SIZE // PIX_PER_CELL) - CELL_PER_BLOCK + 1
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
            if HOG_FEAT == True:
                hog_features = np.hstack((hog_feat1, hog_feat2, hog_feat3))
                features.append(hog_features)

            xleft = xpos*PIX_PER_CELL
            ytop = ypos*PIX_PER_CELL

            # Extract the image patch
            subimg = cv2.resize(ctrans_tosearch[ytop:ytop+WIND_SIZE, xleft:xleft+WIND_SIZE], (64,64))

            # Get color features
            if SPATIAL_FEAT == True:
                spatial_features = bin_spatial(subimg, SPATIAL)
                features.append(spatial_features)
            if HIST_FEAT == True:
                hist_features = color_hist(subimg, NBINS)
                features.append(hist_features)

            # Scale features and make a prediction
            test_features = X_scaler.transform(np.hstack((features)).reshape(1, -1))
            #test_features = X_scaler.transform(np.hstack((shape_feat, hist_feat)).reshape(1, -1))
            test_prediction = svc.predict(test_features)

            if test_prediction == 1:
                xbox_left = np.int(xleft*scale)
                ytop_draw = np.int(ytop*scale)
                win_draw = np.int(window*scale)
                cv2.rectangle(draw_img,(xbox_left, ytop_draw+ystart),(xbox_left+win_draw,ytop_draw+win_draw+ystart),(0,0,255),6)

    return draw_img
