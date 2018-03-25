import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from sklearn.externals import joblib
import numpy as np
import cv2

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

def plot_images(images, cols=2, rows=6, figsize=(15, 13)):
    length = len(images)
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    indexes = range(cols * rows)
    for ax, index in zip(axes.flat, indexes):
        if index < length:
            image = images[index]
            ax.imshow(image)
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
    fig.tight_layout()
    plt.show()

def draw_boxes(img, bboxes, color=(0, 0, 255), thick=3):
    # Make a copy of the image
    draw_img = np.copy(img)
    # Iterate through the bounding boxes
    for bbox in bboxes:
        # Draw a rectangle given bbox coordinates
        cv2.rectangle(draw_img, bbox[0], bbox[1], color, thick)
    return draw_img

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
        cv2.rectangle(img, bbox[0], bbox[1], (255,0,0), 6)
    # Return the image
    return img

def read_image(img_file):
    img = cv2.imread(img_file)
    # Convert to RGB
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def save_model(svc_scaler=(None, None), model_file=None):
    joblib.dump(svc_scaler, model_file)

def load_model(model_file=None):
    clf, scaler = joblib.load(model_file)
    return (clf, scaler)

def convet_color(img, cspace='RGB'):
    if cspace != 'RGB':
        if cspace == 'HSV':
            image = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        elif cspace == 'LUV':
            image = cv2.cvtColor(img, cv2.COLOR_RGB2LUV)
        elif cspace == 'HLS':
            image = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
        elif cspace == 'YUV':
            image = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
        elif cspace == 'YCrCb':
            image = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    else:
        image = np.copy(img)
    return image

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