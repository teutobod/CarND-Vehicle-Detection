import numpy as np
from sklearn.svm import LinearSVC
from sklearn.externals import joblib
from sklearn.preprocessing import StandardScaler


class Classifier:

    def __init__(self, clf=LinearSVC(random_state=0), scaler=None):
        self.clf = clf
        self.scaler = scaler

    def scale(self, X):
        # Create an array stack of feature vectors
        xv = np.vstack(X).astype(np.float64)
        self.scaler = StandardScaler().fit(xv)

    def fit(self,X, y):
        # Apply the scaler to X
        X = self.scaler.transform(X)
        self.clf.fit(X, y)

    def predict(self, X):
        # Apply the scaler to X
        #X = self.scaler.transform(np.array(X).reshape(1, -1))
        X = self.scaler.transform(X)
        return self.clf.predict(X)

    def save(self, file):
        joblib.dump((self.clf, self.scaler), file)

def load_classifier(file):
    (clf, scaler) = joblib.load(file)
    return Classifier(clf, scaler)
