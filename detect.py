import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob
import random
from skimage.feature import hog
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from scipy.ndimage.measurements import label

#---------------------
# Constants
#---------------------
NBINS = 32
COLOR_SPACE = 'RGB'
VIS_HOG = True
ORIENT = 9
PIX_PER_CELL = 8
CELL_PER_BLOCK = 2

#---------------------

cars = glob.glob('car*/*.png')
notcars = glob.glob('non_car*/*.png')

def color_hist(img, nbins, bins_range = (0,256)):
    rhist = np.histogram(img[:,:,0], bins = 32, range = (0, 256))
    ghist = np.histogram(img[:,:,1], bins = 32, range = (0, 256))
    bhist = np.histogram(img[:,:,2], bins = 32, range = (0, 256))

    bin_edges = rhist[1]
    bin_centers = (bin_edges[1:] + bin_edges[0:len(bin_edges) - 1])/2

    hist_features = np.concatenate((rhist[0], ghist[0], bhist[0]))

    return rhist, ghist, bhist, bin_centers, hist_features

def bin_spatial(img, color_space = COLOR_SPACE, size = (32, 32)):
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
    features = cv2.resize(feature_image, size).ravel()
    return features

def get_hog_features(img, orient, pix_per_cell, cell_per_block, vis = False, feature_vec = True):
    if vis == True:
        features, hog_image = hog(img, orientations = orient, pixels_per_cell = (pix_per_cell, pix_per_cell),
                              cells_per_block = (cell_per_block, cell_per_block), transform_sqrt = True,
                              visualise = True, feature_vector = False)
        return features, hog_image
    else:
        features = hog(img, orientations = orient, pixels_per_cel = (pix_per_cell, pix_per_cell),
                   cells_per_block = (cell_per_block, cell_per_block), transform_sqrt = True,
                   visualise = True, feature_vector = feature_vec)
        return features

# ind = np.random.randint(0, len(cars))
#
# image = mpimg.imread(notcars[ind])
# hls_img = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
#
# features, hog_image_h_channel = get_hog_features(hls_img[:,:,0], ORIENT, PIX_PER_CELL, CELL_PER_BLOCK, VIS_HOG, False)
# features, hog_image_l_channel = get_hog_features(hls_img[:,:,1], ORIENT, PIX_PER_CELL, CELL_PER_BLOCK, VIS_HOG, False)
# features, hog_image_s_channel = get_hog_features(hls_img[:,:,2], ORIENT, PIX_PER_CELL, CELL_PER_BLOCK, VIS_HOG, False)

# fig, ax = plt.subplots(nrows = 1, ncols = 4)
# plt.rc('font', size = 5)
# fig.tight_layout()
# plt.subplot(1, 4, 1)
# plt.imshow(image, cmap = 'gray')
# plt.title('Original')
# plt.subplot(1, 4, 2)
# plt.imshow(hog_image_h_channel, cmap = 'gray')
# plt.title('H-Channel HOG Visualization')
# plt.subplot(1, 4, 3)
# plt.imshow(hog_image_l_channel, cmap = 'gray')
# plt.title('L-Channel HOG Visualization')
# plt.subplot(1, 4, 4)
# plt.imshow(hog_image_s_channel, cmap = 'gray')
# plt.title('S-Channel HOG Visualization')
# plt.show()
