import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from sklearn.preprocessing import StandardScaler
from explore_dataset import get_data, split_list
from feature import extract_features

data_samples, labels = get_data()


X_train, X_test = split_list(data_samples, 0.01)

print(len(X_train), len(X_test))

car_features = extract_features(X_train, cspace='RGB', spatial_size=(32, 32),
                        hist_bins=32, hist_range=(0, 256))


print(len(car_features))

# normalize data. Fit scaler on train data and apply scale to all (train and test) data

if len(car_features) > 0:
    # Create an array stack of feature vectors
    X = np.vstack(car_features).astype(np.float64)
    print(len(X[0]))
    # Fit a per-column scaler
    X_scaler = StandardScaler().fit(X)
    # Apply the scaler to X
    scaled_X = X_scaler.transform(X)
    x_ind = np.random.randint(0, len(X_train))

    # Plot an example of raw and scaled features
    fig = plt.figure(figsize=(12,4))
    plt.subplot(131)
    plt.imshow(mpimg.imread(X_train[x_ind]))
    plt.title('Original Image')

    plt.subplot(132)
    plt.plot(X[x_ind])
    plt.title('Raw Features')

    plt.subplot(133)
    plt.plot(scaled_X[x_ind])
    plt.title('Normalized Features')
    fig.tight_layout()
    plt.show()
else:
    print('Your function only returns empty feature vectors...')




# Just for fun choose random car / not-car indices and plot example images
# car_ind = np.random.randint(0, len(cars))
# notcar_ind = np.random.randint(0, len(notcars))

# Read in the image
# image = mpimg.imread(cars[car_ind])
# gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

# # Call our function with vis=True to see an image output
# features, hog_image = get_hog_features(gray, orient= 9,
#                         pix_per_cell= 8, cell_per_block= 2,
#                         vis=True, feature_vec=False)
#
# # Plot the examples
# fig = plt.figure()
# plt.subplot(121)
# plt.imshow(image, cmap='gray')
# plt.title('Example Car Image')
# plt.subplot(122)
# plt.imshow(hog_image, cmap='gray')
# plt.title('HOG Visualization')
# plt.show()