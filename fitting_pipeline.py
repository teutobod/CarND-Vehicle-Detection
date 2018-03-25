import glob
from datetime import datetime
import os
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from helper import save_model, read_image
from features import *

def read_samples(dir, pattern):
    # images = []
    # for dirpath, dirnames, filenames in os.walk(dir):
    #     for dirname in dirnames:
    #         images.append(glob.glob(dir + '/' + dirname + '/' + pattern))
    # flatten = [item for sublist in images for item in sublist]
    # ig = list(map(lambda img: cv2.cvtColor(cv2.imread(img), cv2.COLOR_BGR2RGB), flatten))

    image_files = glob.glob(dir + '/*/' + pattern, recursive=True)
    images = [read_image(i) for i in image_files]
    return images

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

def fit_model(pos_samples, neg_samples, svc, scaler, params):

    # Extracting featured from samples
    t1 = datetime.now()
    pos_features = list(map(lambda img: extract_features(img, params), pos_samples))
    neg_features = list(map(lambda img: extract_features(img, params), neg_samples))
    t2 = datetime.now()
    extraction_time = (t2 - t1).seconds

    # Stacking and scaling
    X = np.vstack((pos_features, neg_features)).astype(np.float64)
    # Defining labels
    pos_labels = np.ones(len(pos_features))
    neg_labels = np.zeros(len(neg_features))
    y = np.hstack((pos_labels, neg_labels))

    # Splitting the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=np.random.randint(0, 100))

    # Scale data based on training set and apply transformation to all data sets (train and test)
    X_scaler = scaler.fit(X_train)

    X_train = X_scaler.transform(X_train)
    X_test = X_scaler.transform(X_test)

    # Fitting the model
    t1 = datetime.now()
    svc.fit(X_train, y_train)
    t2 = datetime.now()
    fit_time = (t2 - t1).seconds

    # Get model accuracy based on test data
    accuracy = svc.score(X_test, y_test)

    return svc, X_scaler, accuracy, (extraction_time, fit_time)


vehicles = read_samples('./data/vehicles', '*.png')
no_vehicles = read_samples('./data/non-vehicles', '*.png')

params = FeatureParameter()
clf_model, scaler, model_accuracy, times = fit_model(vehicles, no_vehicles, LinearSVC(), StandardScaler(), params)

print('Feature extraction time: {} seconds'.format(times[0]))
print('Fitting time: {} seconds'.format(times[1]))
print('Model accuracy: {}'.format(model_accuracy))

model_file = "./data/model.p"
print('Saving classifier model to file', model_file)
save_model((clf_model, scaler), model_file)
