#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import LinearSVC
from util.feature_extractor import Features
from sklearn.model_selection import cross_val_score


def test_params():
    examples = [1000, 2000, 5000, 10000, None]
    clusters = [10, 20, 30, 50, 100]
    features = [50, 100, 150, 200, 300]

    accuracy_ex = []
    accuracy_cl = []
    accuracy_ft = []

    for i, ex in enumerate(examples):
        print "Testing with examples", ex
        try:
            f = Features()
            X, y = f.get_orb_train("data/tinyX.npy", "data/tinyY.npy",
                                   max_examples=ex, n_clusters=30, n_features=100)

            svm = LinearSVC()
            accuracy_ex.append(cross_val_score(svm, X, y).mean())
            print accuracy_ex[-1]
        except MemoryError:
            print "Memory error for", i, "Passing"

    for i, cluster in enumerate(clusters):
        print "Testing with n_clusters", cluster
        try:
            f = Features()
            X, y = f.get_orb_train("data/tinyX.npy", "data/tinyY.npy",
                                   max_examples=5000, n_clusters=cluster, n_features=100)

            svm = LinearSVC()
            accuracy_cl.append(cross_val_score(svm, X, y).mean())
        except MemoryError:
            print "Memory error for", i, "Passing"

    for i, feat in enumerate(features):
        print "Testing with features", feat
        try:
            f = Features()
            X, y = f.get_orb_train("data/tinyX.npy", "data/tinyY.npy",
                                   max_examples=5000, n_clusters=30, n_features=feat)

            svm = LinearSVC()
            accuracy_ft.append(cross_val_score(svm, X, y).mean())
        except MemoryError:
            print "Memory error for", i, "Passing"

    # PLOT EXAMPLES.
    print max(accuracy_ex), examples[np.argmax(accuracy_ex)]
    plt.figure(1)

    plt.title("Accuracy vs Number of Examples")
    plt.xlabel("Number of examples")
    plt.ylabel("Accuracy")
    plt.plot(examples, accuracy_ex, '-ro')

    # PLOT CLUSTERS.
    print max(accuracy_cl), clusters[np.argmax(accuracy_cl)]
    plt.figure(2)

    plt.title("Accuracy vs Number of Clusters")
    plt.xlabel("Number of clusters")
    plt.ylabel("Accuracy")
    plt.plot(clusters, accuracy_cl, '-ro')

    # PLOT FEATURES.
    print max(accuracy_ft), features[np.argmax(accuracy_ft)]
    plt.figure(3)

    plt.title("Accuracy vs Maximum Features")
    plt.xlabel("Maximum number of features")
    plt.ylabel("Accuracy")
    plt.plot(features, accuracy_ft, '-ro')

    plt.show()


if __name__ == '__main__':
    test_params()
