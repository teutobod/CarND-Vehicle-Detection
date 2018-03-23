import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

def plot_random_example(X_train):
    # Plot an example of raw and scaled features
    x_ind = np.random.randint(0, len(X_train))
    fig = plt.figure(figsize=(12, 4))
    plt.subplot(131)
    plt.imshow(mpimg.imread(X_train[x_ind]))
    plt.title('Original Image')

    plt.subplot(132)
    plt.plot(X[x_ind])
    plt.title('Raw Features')

    # plt.subplot(133)
    # plt.plot(scaled_X[x_ind])
    # plt.title('Normalized Features')
    # fig.tight_layout()
    # plt.show()

def plot_random_results(X, y):
    fig = plt.figure(figsize=(12, 4))

    for i in range(9):
        plt.subplot(330 + i + 1)

        x_i = np.random.randint(0, len(X))
        plt.imshow(mpimg.imread(X[x_i]))
        plt.title('Sample image with result {}'.format(y[x_i]))

    fig.tight_layout()
    plt.show()



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