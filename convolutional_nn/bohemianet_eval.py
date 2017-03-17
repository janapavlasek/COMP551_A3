"""Evaluate a convolutional neural network on the validation set
"""
import cv2
import time
import numpy as np
from bohemianet import Bohemianet
from bohemianet_input import get_valid_batches


if __name__ == "__main__":
    bohemianet = Bohemianet(model_name="model1")
    bohemianet.load()

    acc = bohemianet.validate(conf_m=True)

    print("Accuracy: {:.1%}".format(acc))
