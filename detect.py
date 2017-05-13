import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob
import random
import time
from skimage.feature import hog
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from scipy.ndimage.measurements import label

#---------------------
# Constants
#---------------------
NBINS = 32
SPATIAL = 32
COLOR_SPACE = 'RGB'
VIS_HOG = False
FEATURE_VECTOR = True
ORIENT = 9
PIX_PER_CELL = 8
CELL_PER_BLOCK = 2
HOG_CHANNEL = 0

#---------------------

cars = glob.glob('car*/*.png')
notcars = glob.glob('non_car*/*.png')

def color_hist(img, nbins, bins_range = (0,256)):
    channel1_hist = np.histogram(img[:,:,0], bins = nbins, range = bins_range)
    channel2_hist = np.histogram(img[:,:,1], bins = nbins, range = bins_range)
    channel3_hist = np.histogram(img[:,:,2], bins = nbins, range = bins_range)

    hist_features = np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))

    return hist_features

def bin_spatial(img, size = (32, 32)):
    features = cv2.resize(img, size).ravel()
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
    features = []
    for img in imgs:
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
        spatial_features = bin_spatial(feature_image, (SPATIAL, SPATIAL))
        color_features = color_hist(feature_image, NBINS, (0, 256))
        # Call get_hog_features() with vis=False, feature_vec=True
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
        features.append(np.concatenate((spatial_features, color_features, hog_features)))
    return features


car_features = extract_features(cars, COLOR_SPACE, ORIENT, PIX_PER_CELL, CELL_PER_BLOCK, HOG_CHANNEL)
notcar_features = extract_features(notcars, COLOR_SPACE, ORIENT, PIX_PER_CELL, CELL_PER_BLOCK, HOG_CHANNEL)

X = np.vstack((car_features, notcar_features)).astype(np.float64)
X_scaler = StandardScaler().fit(X)
scaled_X = X_scaler.transform(X)

y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))

rand_state = np.random.randint(0, 100)
X_train, X_test, y_train, y_test = train_test_split(scaled_X, y, test_size = 0.2, random_state = rand_state)

print('Using spatial binning of:', SPATIAL,
    'and', NBINS,'histogram bins with', ORIENT,
    'orientations', PIX_PER_CELL, 'pixels per cell and',
    CELL_PER_BLOCK, 'cells per block')
print('Feature vector length:', len(X_train[0]))
# Use a linear SVC
svc = LinearSVC()
# Check the training time for the SVC
t=time.time()
svc.fit(X_train, y_train)
t2 = time.time()
print(round(t2-t, 2), 'Seconds to train SVC...')
# Check the score of the SVC
print('Test Accuracy of SVC = ', round(svc.score(X_test, y_test), 4))
# Check the prediction time for a single sample
t=time.time()
n_predict = 10
print('My SVC predicts: ', svc.predict(X_test[0:n_predict]))
print('For these',n_predict, 'labels: ', y_test[0:n_predict])
t2 = time.time()
print(round(t2-t, 5), 'Seconds to predict', n_predict,'labels with SVC')

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
