import numpy as np
import cv2
import glob
from datetime import datetime
import os
from sklearn.svm import LinearSVC
from helper import save_model
from feature import *

def readImages(dir, pattern):
    """
    Returns an image list with the image contained on the directory `dir` matching the `pattern`.
    """
    images = []
    for dirpath, dirnames, filenames in os.walk(dir):
        for dirname in dirnames:
            images.append(glob.glob(dir + '/' + dirname + '/' + pattern))
    flatten = [item for sublist in images for item in sublist]
    return list(map(lambda img: cv2.cvtColor(cv2.imread(img), cv2.COLOR_BGR2RGB), flatten))

# def bin_spatial(img, size=(32, 32)):
#     # Use cv2.resize().ravel() to create the feature vector
#     features = cv2.resize(img, size).ravel()
#     # Return the feature vector
#     return features


# def color_hist(img, nbins=32, bins_range=(0, 256)):
#     # Compute the histogram of the color channels separately
#     channel1_hist = np.histogram(img[:, :, 0], bins=nbins, range=bins_range)
#     channel2_hist = np.histogram(img[:, :, 1], bins=nbins, range=bins_range)
#     channel3_hist = np.histogram(img[:, :, 2], bins=nbins, range=bins_range)
#     # Concatenate the histograms into a single feature vector
#     hist_features = np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))
#     # Return the individual histograms, bin_centers and feature vector
#     return hist_features


# from skimage.feature import hog
# def get_hog_features(img, orient, pix_per_cell, cell_per_block,
#                      vis=False, feature_vec=True):
#     # Call with two outputs if vis==True
#     if vis == True:
#         features, hog_image = hog(img, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell),
#                                   cells_per_block=(cell_per_block, cell_per_block), transform_sqrt=True,
#                                   visualise=vis, feature_vector=feature_vec)
#         return features, hog_image
#     # Otherwise call with one output
#     else:
#         features = hog(img, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell),
#                        cells_per_block=(cell_per_block, cell_per_block), transform_sqrt=True,
#                        visualise=vis, feature_vector=feature_vec)
#         return features


# Value object to hold all feature extraction parameters.
# class FeatureParameter():
#     def __init__(self):
#         self.cspace = 'YCrCb'
#         self.size = (16, 16)
#         self.hist_bins = 32
#         self.hist_range = (0, 256)
#         self.orient = 8
#         self.pix_per_cell = 8
#         self.cell_per_block = 2
#         self.hog_channel = 'ALL'


# def extract_features(image, params):
#     # Parameters extraction
#     # HOG parameters
#     cspace = params.cspace
#     orient = params.orient
#     pix_per_cell = params.pix_per_cell
#     cell_per_block = params.cell_per_block
#     hog_channel = params.hog_channel
#     # Spatial parameters
#     size = params.size
#     # Histogram parameters
#     hist_bins = params.hist_bins
#     hist_range = params.hist_range
#
#     # apply color conversion if other than 'RGB'
#     if cspace != 'RGB':
#         if cspace == 'HSV':
#             feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
#         elif cspace == 'LUV':
#             feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2LUV)
#         elif cspace == 'HLS':
#             feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
#         elif cspace == 'YUV':
#             feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
#         elif cspace == 'YCrCb':
#             feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
#     else:
#         feature_image = np.copy(image)
#
#     # Call get_hog_features() with vis=False, feature_vec=True
#     if hog_channel == 'ALL':
#         hog_features = []
#         for channel in range(feature_image.shape[2]):
#             hog_features.append(get_hog_features(feature_image[:, :, channel],
#                                                  orient, pix_per_cell, cell_per_block,
#                                                  vis=False, feature_vec=True))
#         hog_features = np.ravel(hog_features)
#     else:
#         hog_features = get_hog_features(feature_image[:, :, hog_channel], orient,
#                                         pix_per_cell, cell_per_block, vis=False, feature_vec=True)
#
#     # Apply bin_spatial() to get spatial color features
#     spatial_features = bin_spatial(feature_image, size)
#
#     # Apply color_hist()
#     hist_features = color_hist(feature_image, nbins=hist_bins, bins_range=hist_range)
#
#     return np.concatenate((spatial_features, hist_features, hog_features))

def get_data():
    cars = []
    notcars = []

    images = glob.glob('data/*/*/*.png', recursive=True)
    for image in images:
        if 'non-vehicles' in image:
            notcars.append(image)
        else:
            cars.append(image)
    return cars, notcars


from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


def fitModel(positive, negative, svc, scaler, params):

    positive_features = list(map(lambda img: extract_features(img, params), positive))
    negatice_features = list(map(lambda img: extract_features(img, params), negative))

    # Stacking and scaling
    X = np.vstack((positive_features, negatice_features)).astype(np.float64)
    # Defining labels
    y = np.hstack((np.ones(len(positive_features)), np.zeros(len(negatice_features))))

    # Split up data
    rand_state = np.random.randint(0, 100)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=rand_state)

    # Scale data based on training set and apply transformation to all data sets (train and test)
    X_scaler = scaler.fit(X_train)

    X_train = X_scaler.transform(X_train)
    X_test = X_scaler.transform(X_test)

    # Fitting
    t1 = datetime.now()
    svc.fit(X_train, y_train)
    t2 = datetime.now()

    accuracy = svc.score(X_test, y_test)

    return (svc, X_scaler, (t2-t1).seconds, accuracy)


vehicles = readImages('./data/vehicles', '*.png')
non_vehicles = readImages('./data/non-vehicles', '*.png')

params = FeatureParameter()
svc, scaler, fitting_duration, accuracy = fitModel(vehicles, non_vehicles, LinearSVC(), StandardScaler(), params)
print('Fitting time: {} sec, Model accuracy: {}'.format(fitting_duration, accuracy))

model_file = "model.p"
print('Saving classifier model to file', model_file)
save_model((svc, scaler), model_file)
