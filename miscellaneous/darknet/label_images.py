#!/usr/bin/env python
import os
import cv2
import sys
import numpy as np

if len(sys.argv) < 4:
    sys.exit("Usage: python label_images.py PATH_TO_Y PATH_TO_X PATH_TO_X_TEST IMAGE_PATH")

Y_PATH = sys.argv[1]
# y = np.load(Y_PATH)
size = 64

X_PATH = sys.argv[2]
# X = np.load(X_PATH)

X_OUT_PATH = sys.argv[3]
X_OUT = np.load(X_OUT_PATH)

IMAGE_PATH = sys.argv[4]

print "Using label file", Y_PATH, "with input", X_PATH
print "Saving images to", IMAGE_PATH

# with open("yolo_a3.names", "w") as f:
#     for i in range(0, 40):
#         f.write("{}\n".format(i))

# with open("train.txt", "w") as f:
#     for i in range(0, X.shape[0]):
#         img_path = "image_{}.jpg".format(i)
#         cv2.imwrite(os.path.join(IMAGE_PATH, img_path), X[i].T)
#         f.write("{}\n".format(os.path.join(IMAGE_PATH, img_path)))

#         label_file = "image_{}.txt".format(i)

#         with open(os.path.join(IMAGE_PATH, label_file), "w") as g:
#             g.write("{} {} {} {} {}\n".format(y[i], 0, 0, size, size))

for i in range(0, X_OUT.shape[0]):
    img_path = "image_{}.jpg".format(i)
    cv2.imwrite(os.path.join(IMAGE_PATH, "test", img_path), X_OUT[i].T)
