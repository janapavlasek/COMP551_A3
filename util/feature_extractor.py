#!/usr/bin/env python
import cv2
import warnings
import numpy as np
from random import shuffle
from sklearn.cluster import KMeans
from scipy.sparse import csr_matrix


class Features(object):
    """Features class."""
    NUM_COLS = 64 * 64 * 3

    def __init__(self):
        self.orb = None
        self.kmeans = None
        self.n_clusters = None

    def get_orb_train(self, x_train_file, y_train_file, threshold=10,
                      n_clusters=20, n_features=100, max_examples=None):
        X = np.load(x_train_file)
        y = np.load(y_train_file)

        n_samples = X.shape[0]

        # Shuffle the data.
        data = zip(X, y)
        shuffle(data)

        # Return the data to the original shape.
        for i in range(0, n_samples):
            X[i] = data[i][0]
            y[i] = data[i][1]

        if max_examples is not None:
            X = X[0:max_examples]
            y = y[0:max_examples]

        n_samples = X.shape[0]

        self.n_clusters = n_clusters

        # nfeatures=500, scaleFactor=1.2, nlevels=8, edgeThreshold=31,
        # firstLevel=0, WTA_K=2, scoreType=ORB::HARRIS_SCORE, patchSize=31
        self.orb = cv2.ORB(n_features, 1.2, 8, threshold, 0, 2)

        print "Getting keypoint descriptors."

        # This array contains all the keypoint descriptions in all the images,
        # of shape (n_keypts, n_features).
        keypts = []
        ignored = 0

        for i in range(0, n_samples):
            kp = self.orb.detect(X[i].T, None)

            kp, des = self.orb.compute(X[i].T, kp)

            if len(kp) < 1:
                ignored += 1
                continue

            keypts += des.tolist()

        print ignored, "images were ignored since keypoints were not found."

        print "Classifying the descriptors."

        # Classify the keypoint descriptors.
        self.kmeans = KMeans(n_clusters=n_clusters)
        self.kmeans.fit(keypts)

        warnings.filterwarnings('ignore')

        print "Getting bag of words."

        # The bag of words feature matrix.
        features = csr_matrix((n_samples, n_clusters))

        # We can now classify each keypoint in a "bag of words" type feature.
        for i in range(0, n_samples):
            # Get the keypoints.
            kp = self.orb.detect(X[i].T, None)
            kp, des = self.orb.compute(X[i].T, kp)

            try:
                # Get the keypoint descriptors.
                classes = self.kmeans.predict(des)
            except ValueError:
                print "Value error for example", i, "- skipping."
                continue

            for j in range(0, len(classes)):
                features[i, classes[j]] += 1

        return features, y

    def get_orb_in(self, x_in_file):
        if self.orb is None or self.kmeans is None:
            print "You need to get the training data before the input data."
            return

        X = np.load(x_in_file)

        # The bag of words feature matrix.
        features = np.zeros((X.shape[0], self.n_clusters))

        # We can now classify each keypoint in a "bag of words" type feature.
        for i in range(0, X.shape[0]):
            # Get the keypoints.
            kp = self.orb.detect(X[i].T, None)
            kp, des = self.orb.compute(X[i].T, kp)

            # Get the keypoint descriptors.
            classes = self.kmeans.predict(des)

            for j in range(0, len(classes)):
                features[i, classes[j]] += 1

        return X

    def get_pixels_train(self, x_train_file, y_train_file):
        X = np.load(x_train_file)
        y = np.load(y_train_file)

        # Shuffle the data.
        data = zip(X, y)
        shuffle(data)

        # Return the data to the original shape.
        for i in range(0, X.shape[0]):
            X[i] = data[i][0]
            y[i] = data[i][1]

        X = np.reshape(X, (X.shape[0], self.NUM_COLS))

        return X, y

    def get_pixels_in(self, x_in_file):
        X = np.load(x_in_file)
        X = np.reshape(X, (X.shape[0], self.NUM_COLS))

        return X


if __name__ == '__main__':
    features = Features()
    features.get_orb_train("data/tinyX.npy", "data/tinyY.npy")
