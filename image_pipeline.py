import glob
from feature import *
from helper import *


def slide_window(img, x_start_stop=[None, None], y_start_stop=[None, None],
                 xy_window=(64, 64), xy_overlap=(0.5, 0.5)):
    # If x and/or y start/stop positions not defined, set to image size
    if x_start_stop[0] == None:
        x_start_stop[0] = 0
    if x_start_stop[1] == None:
        x_start_stop[1] = img.shape[1]
    if y_start_stop[0] == None:
        y_start_stop[0] = 0
    if y_start_stop[1] == None:
        y_start_stop[1] = img.shape[0]
    # Compute the span of the region to be searched
    xspan = x_start_stop[1] - x_start_stop[0]
    yspan = y_start_stop[1] - y_start_stop[0]
    # Compute the number of pixels per step in x/y
    nx_pix_per_step = np.int(xy_window[0 ] *(1 - xy_overlap[0]))
    ny_pix_per_step = np.int(xy_window[1 ] *(1 - xy_overlap[1]))
    # Compute the number of windows in x/y
    nx_buffer = np.int(xy_window[0 ] *(xy_overlap[0]))
    ny_buffer = np.int(xy_window[1 ] *(xy_overlap[1]))
    nx_windows = np.int((xspan - nx_buffer ) /nx_pix_per_step)
    ny_windows = np.int((yspan - ny_buffer ) /ny_pix_per_step)
    # Initialize a list to append window positions to
    window_list = []
    # Loop through finding x and y window positions
    # Note: you could vectorize this step, but in practice
    # you'll be considering windows one by one with your
    # classifier, so looping makes sense
    for ys in range(ny_windows):
        for xs in range(nx_windows):
            # Calculate window position
            startx = xs * nx_pix_per_step + x_start_stop[0]
            endx = startx + xy_window[0]
            starty = ys * ny_pix_per_step + y_start_stop[0]
            endy = starty + xy_window[1]
            # Append window position to list
            window_list.append(((startx, starty), (endx, endy)))
    return window_list

def find_cars_window(img, clf, scaler, params, y_start_stop=[360, 700], xy_window=(64, 64), xy_overlap=(0.85, 0.85)):
    # 1) Create an empty list to receive positive detection windows
    car_windows = []
    # 2) Get search windows
    windows = slide_window(img, y_start_stop=y_start_stop, xy_window=xy_window, xy_overlap=xy_overlap)
    # 3) Iterate over all windows in the list
    for window in windows:
        # 3) Extract the test window from original image
        img_window = cv2.resize(img[window[0][1]:window[1][1], window[0][0]:window[1][0]], (64, 64))
        # 4) Extract features for that window
        features = extract_features(img_window, params)
        # 5) Scale extracted features
        scaled_features = scaler.transform(features.reshape(1, -1))
        # 6) Predict with classifier
        prediction = clf.predict(scaled_features)
        # 7) If positive save the window
        if prediction == 1:
            car_windows.append(window)
    return car_windows


def find_cars(img, clf, scaler, params, y_start_stop=[350, 656], window=64, cells_per_step=1, scale=1.5):
    cspace = params.cspace
    size = params.size
    hist_bins = params.hist_bins
    hist_range = params.hist_range
    orient = params.orient
    pix_per_cell = params.pix_per_cell
    cell_per_block = params.cell_per_block

    # Apply color conversion if needed
    feature_image = convet_color(img, cspace)

    ystart, ystop = y_start_stop
    ctrans_tosearch = feature_image[ystart:ystop, :, :]
    if scale != 1:
        imshape = ctrans_tosearch.shape
        ctrans_tosearch = cv2.resize(ctrans_tosearch, (np.int(imshape[1] / scale), np.int(imshape[0] / scale)))

    ch1 = ctrans_tosearch[:, :, 0]
    ch2 = ctrans_tosearch[:, :, 1]
    ch3 = ctrans_tosearch[:, :, 2]

    # Define blocks and steps as above
    nxblocks = (ch1.shape[1] // pix_per_cell) - cell_per_block + 1
    nyblocks = (ch1.shape[0] // pix_per_cell) - cell_per_block + 1
    nfeat_per_block = orient * cell_per_block ** 2

    nblocks_per_window = (window // pix_per_cell) - cell_per_block + 1
    nxsteps = (nxblocks - nblocks_per_window) // cells_per_step
    nysteps = (nyblocks - nblocks_per_window) // cells_per_step

    # Compute individual channel HOG features for the entire image
    hog1 = get_hog_features(ch1, orient, pix_per_cell, cell_per_block, feature_vec=False)
    hog2 = get_hog_features(ch2, orient, pix_per_cell, cell_per_block, feature_vec=False)
    hog3 = get_hog_features(ch3, orient, pix_per_cell, cell_per_block, feature_vec=False)

    car_windows = []

    for xb in range(nxsteps):
        for yb in range(nysteps):
            ypos = yb * cells_per_step
            xpos = xb * cells_per_step

            # Extract HOG features
            hog_feat1 = hog1[ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel()
            hog_feat2 = hog2[ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel()
            hog_feat3 = hog3[ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel()

            hog_features = np.hstack((hog_feat1, hog_feat2, hog_feat3))

            xleft = xpos * pix_per_cell
            ytop = ypos * pix_per_cell

            # Extract the image patch
            subimg = cv2.resize(ctrans_tosearch[ytop:ytop + window, xleft:xleft + window], (64, 64))

            # Get color features
            spatial_features = bin_spatial(subimg, size=size)
            hist_features = color_hist(subimg, nbins=hist_bins, bins_range=hist_range)

            # Scale features and make a prediction
            sclaed_features = scaler.transform(np.hstack((spatial_features, hist_features, hog_features)).reshape(1, -1))
            prediction = clf.predict(sclaed_features)

            if prediction == 1:
                xbox_left = np.int(xleft * scale)
                ytop_draw = np.int(ytop * scale)
                win_draw = np.int(window * scale)

                startx = xbox_left
                starty = ytop_draw + ystart
                endx = xbox_left + win_draw
                endy = ytop_draw + win_draw + ystart

                box = ((startx, starty), (endx, endy))

                car_windows.append(box)
    return car_windows

def drawCars(img, windows):
    output = np.copy(img)
    return draw_boxes(output, windows)


test_images = list(map(lambda img: read_image(img), glob.glob('./test_images/*.jpg')))

model_file = 'model.p'
print('Loading classifier model from file', model_file)
clf, scaler = load_model(model_file)


parameter = FeatureParameter()

# car_on_test = list(map(lambda img: drawCars(img, findCarWindows(img, clf, scaler, parameter), test_images))
# fast_boxes = list(map(lambda img: find_cars(img, clf, scaler, parameter), test_images))
# fast_on_test = list(map(lambda imgAndBox: drawCars(imgAndBox[0], imgAndBox[1]), zip(test_images, fast_boxes)))

boxes = []
box_imgs = []
label_box_imgs = []
for img in test_images:
    car_boxes = find_cars(img, clf, scaler, parameter)

    boxes.append(car_boxes)

    box_imgs.append(drawCars(img, car_boxes))

    from heatmap import HeatMap

    heatmap = HeatMap(threshold=2)
    heatmap.add_heat(car_boxes)
    heatmap.apply_threshold()

    from scipy.ndimage.measurements import label
    labels = label(heatmap.get_headmap())

    label_box_img = draw_labeled_bboxes(np.copy(img), labels)
    box_imgs.append(label_box_img)

#plot_images(car_on_test)
plot_images(box_imgs)
#plot_images(label_box_imgs)


