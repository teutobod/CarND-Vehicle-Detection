import numpy as np
import matplotlib.image as mpimg
import glob

import random

# Define a function to return some characteristics of the dataset
def data_look(car_list, notcar_list):
    data_dict = {}
    # Define a key in data_dict "n_cars" and store the number of car images
    data_dict["n_cars"] = len(car_list)
    # Define a key "n_notcars" and store the number of notcar images
    data_dict["n_notcars"] = len(notcar_list)
    # Read in a test image, either car or notcar
    example_img = mpimg.imread(car_list[0])
    # Define a key "image_shape" and store the test image shape 3-tuple
    data_dict["image_shape"] = example_img.shape
    # Define a key "data_type" and store the data type of the test image.
    data_dict["data_type"] = example_img.dtype
    # Return data_dict
    return data_dict


def get_data():
    cars = []
    notcars = []

    images = glob.glob('data/*/*/*.png', recursive=True)
    for image in images:
        if 'non-vehicles' in image:
            notcars.append(image)
        else:
            cars.append(image)

    # Get same number of samples from each class
    min_len = min(len(cars), len(notcars))

    cars = np.asarray(cars[:min_len])
    notcars = np.asarray(notcars[:min_len])
    samples = np.concatenate((cars, notcars), axis=0)

    random.shuffle(samples)
    # Create labels 1 for cars 0 for notcars
    labels = np.ones_like(samples, dtype=np.int16)
    mask = ['non-vehicles' in d for d in samples]
    labels[mask] = 0
    return samples, labels


def explore_data():
    cars, notcars = get_data()
    data_info = data_look(cars, notcars)

    print('Your function returned a count of',
          data_info["n_cars"], ' cars and',
          data_info["n_notcars"], ' non-cars')
    print('of size: ', data_info["image_shape"], ' and data type:',
          data_info["data_type"])
