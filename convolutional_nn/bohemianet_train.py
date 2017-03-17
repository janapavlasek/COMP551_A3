"""Train a convolutional neural network
"""
from bohemianet import Bohemianet


if __name__ == "__main__":
    bohemianet = Bohemianet(model_name="model1")
    # bohemianet.load()
    bohemianet.optimize(
        num_iterations=400001, batch_size=64, learning_rate=1e-4, augment=False)
