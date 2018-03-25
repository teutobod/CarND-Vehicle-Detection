import numpy as np
from datetime import datetime

from sklearn.model_selection import train_test_split

from explore_dataset import get_data
from features import extract_features
from classifier import Classifier

data_samples, labels = get_data()

X_train, X_test, y_train, y_test = train_test_split(data_samples, labels, test_size=0.2, shuffle=True)
print(len(X_train), len(X_test))
print(len(y_train), len(y_test))
#hp.plot_random_results(X_train, y_train)

print("Started feature extraction ...")
t1 = datetime.now()

color_space = 'YCrCb'  # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
spatial_size = (16, 16)  # Spatial binning dimensions
hist_bins = 32    # Number of histogram bins
orient = 8  # HOG orientations
#hog_channel = 0  # Can be 0, 1, 2, or "ALL"

X_train_features = extract_features(X_train, color_space, spatial_size, hist_bins, orient)
X_test_features = extract_features(X_test, color_space, spatial_size, hist_bins, orient)
t2 = datetime.now()
print("Feature extraction took: ", (t2-t1).seconds, "seconds")
print("Length of feature vector:  ", len(X_train_features[0]))

# Training
print("Started training ...")
t1 = datetime.now()
clf = Classifier()
clf.scale(X_train_features)
clf.fit(X_train_features, y_train)
t2 = datetime.now()
print("Training took: ", (t2-t1).seconds, "seconds")

# Save the trained model
clf.save('clf_model.p')

# Evaluate
prediction = clf.predict(X_test_features)

# Plot some random prediction results
# hp.plot_random_results(X_test, prediction)
# hp.plot_random_results(X_test, prediction)
# hp.plot_random_results(X_test, prediction)
# hp.plot_random_results(X_test, prediction)

from sklearn.metrics import accuracy_score, classification_report
acc = accuracy_score(y_test, prediction)
print("Accuracy is: ", acc)

target_names = ['no cars', 'car']
report = classification_report(y_test, prediction, target_names=target_names)
print(report)
