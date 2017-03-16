#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from util.feature_extractor import Features
from sklearn.cross_validation import train_test_split
from sklearn.model_selection import cross_val_score


def score(y_pred, y_expect):
    errors = 0.0
    for i in range(0, len(y_pred)):
        if y_pred[i] != y_expect[i]:
            errors += 1.0

    return 1 - errors / float(len(y_pred))


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

            log_reg = LogisticRegression()
            accuracy_ex.append(cross_val_score(log_reg, X, y).mean())
        except MemoryError:
            print "Memory error for", i, "Passing"

    for i, cluster in enumerate(clusters):
        print "Testing with n_clusters", cluster
        try:
            f = Features()
            X, y = f.get_orb_train("data/tinyX.npy", "data/tinyY.npy",
                                   max_examples=5000, n_clusters=cluster, n_features=100)

            log_reg = LogisticRegression()
            accuracy_cl.append(cross_val_score(log_reg, X, y).mean())
        except MemoryError:
            print "Memory error for", i, "Passing"

    for i, feat in enumerate(features):
        print "Testing with features", feat
        try:
            f = Features()
            X, y = f.get_orb_train("data/tinyX.npy", "data/tinyY.npy",
                                   max_examples=5000, n_clusters=30, n_features=feat)

            log_reg = LogisticRegression()
            accuracy_ft.append(cross_val_score(log_reg, X, y).mean())
        except MemoryError:
            print "Memory error for", i, "Passing"

    # PLOT EXAMPLES.
    print min(accuracy_ex), examples[np.argmin(accuracy_ex)]
    plt.figure(1)

    plt.title("Accuracy vs Number of Examples")
    plt.xlabel("Number of examples")
    plt.ylabel("Accuracy")
    plt.plot(examples, accuracy_ex, '-ro')

    # PLOT CLUSTERS.
    print min(accuracy_cl), clusters[np.argmin(accuracy_cl)]
    plt.figure(2)

    plt.title("Accuracy vs Number of Clusters")
    plt.xlabel("Number of clusters")
    plt.ylabel("Accuracy")
    plt.plot(clusters, accuracy_cl, '-ro')

    # PLOT FEATURES.
    print min(accuracy_ft), features[np.argmin(accuracy_ft)]
    plt.figure(3)

    plt.title("Accuracy vs Maximum Features")
    plt.xlabel("Maximum number of features")
    plt.ylabel("Accuracy")
    plt.plot(features, accuracy_ft, '-ro')

    plt.show()


if __name__ == '__main__':
    test_params()
    # print "Getting features"
    # features = Features()
    # X, y = features.get_orb_train("data/tinyX.npy", "data/tinyY.npy", max_examples=5000,
    #                               n_clusters=50, n_features=100)
    # # X, y = features.get_pixels_train("data/tinyX.npy", "data/tinyY.npy")

    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    # print "Training"
    # log_reg = LogisticRegression()
    # log_reg.fit(X_train, y_train)

    # print "Predicting"

    # y_pred = log_reg.predict(X_test)

    # print "Scoring"

    # print "Score:", score(y_pred, y_test)

    # print "Cross validation score:", cross_val_score(log_reg, X, y).mean()
