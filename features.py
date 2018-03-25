import numpy as np
import cv2
from skimage.feature import hog
from helper import convet_color

class FeatureParameter():
    def __init__(self):
        self.cspace = 'YCrCb'

        self.size = (16, 16)
        self.hist_bins = 32
        self.hist_range = (0, 256)

        self.hog_channel = 'ALL'
        self.orient = 8
        self.pix_per_cell = 8
        self.cell_per_block = 2

def bin_spatial(img, size=(32, 32)):
    # Use cv2.resize().ravel() to create the feature vector
    features = cv2.resize(img, size).ravel()
    # Return the feature vector
    return features

def color_hist(img, nbins=32, bins_range=(0, 256)):
    # Compute the histogram of the color channels separately
    channel1_hist = np.histogram(img[:, :, 0], bins=nbins, range=bins_range)
    channel2_hist = np.histogram(img[:, :, 1], bins=nbins, range=bins_range)
    channel3_hist = np.histogram(img[:, :, 2], bins=nbins, range=bins_range)
    # Concatenate the histograms into a single feature vector
    hist_features = np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))
    # Return the individual histograms, bin_centers and feature vector
    return hist_features

def get_hog_features(img, orient, pix_per_cell, cell_per_block, vis=False, feature_vec=True):
    def my_hog():
        return hog(img, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell),
                   cells_per_block=(cell_per_block, cell_per_block), transform_sqrt=True,
                   visualise=vis, feature_vector=feature_vec)

    if vis == True:
        features, hog_image = my_hog()
        return features, hog_image
    else:
        features = my_hog()
        return features

def extract_features(image, params):
    cspace = params.cspace

    size = params.size
    hist_bins = params.hist_bins
    hist_range = params.hist_range

    hog_channel = params.hog_channel
    orient = params.orient
    pix_per_cell = params.pix_per_cell
    cell_per_block = params.cell_per_block

    # Apply color conversion
    feature_image = convet_color(image, cspace)

    # Call get_hog_features() with vis=False, feature_vec=True
    if hog_channel == 'ALL':
        hog_features = []
        for channel in range(feature_image.shape[2]):
            hog_features.append(get_hog_features(feature_image[:, :, channel], orient, pix_per_cell, cell_per_block))
        hog_features = np.ravel(hog_features)
    else:
        hog_features = get_hog_features(feature_image[:, :, hog_channel], orient, pix_per_cell, cell_per_block)

    # Apply bin_spatial
    spatial_features = bin_spatial(feature_image, size)

    # Apply color_histogramm
    hist_features = color_hist(feature_image, nbins=hist_bins, bins_range=hist_range)

    feature_vector = np.concatenate((spatial_features, hist_features, hog_features))
    return feature_vector

# def single_img_features(img, cspace, spatial_size, hist_bins, orient):
#     # Apply color conversion
#     feature_image = convet_color(img, cspace)
#
#     # Apply bin_spatial() to get spatial color features
#     spatial_features = bin_spatial(feature_image, size=spatial_size)
#
#     # Apply color_hist() also with a color space option now
#     hist_features = color_hist(feature_image, nbins=hist_bins)
#
#     # Apply HOG features
#     hog_features = []
#     for channel in range(feature_image.shape[2]):
#         img_ch = feature_image[:, :, channel]
#         hog_features.append(get_hog_features(img_ch, orient=orient))
#     hog_features = np.ravel(hog_features)
#
#     # Append the new feature vector to the features list
#     feature_vec = np.concatenate((spatial_features, hist_features, hog_features))
#     # feature_vec = np.concatenate((hist_features, hog_features))
#     # feature_vec = hog_features
#     return feature_vec
#
#
# # Function to extract features from a list of images
# def extract_features(imgs, cspace, spatial_size, hist_bins, orient):
#     # Create a list to append feature vectors to
#     features = []
#     for file in imgs:
#         #image = mpimg.imread(file)
#         image = cv2.cvtColor(cv2.imread(file), cv2.COLOR_BGR2RGB)
#         feature_vec = single_img_features(image, cspace, spatial_size, hist_bins, orient)
#         features.append(feature_vec)
#     return features
