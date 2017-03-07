#!/usr/bin/env python
from sklearn.linear_model import LogisticRegression
from util.feature_extractor import Features
from sklearn.cross_validation import cross_val_score


if __name__ == '__main__':
    features = Features()
    X, y = features.get_orb_train("data/tinyX.npy", "data/tinyY.npy")

    log_reg = LogisticRegression()
    # log_reg.fit(X, y)

    print "Cross validation score:", cross_val_score(log_reg, X, y).mean()
