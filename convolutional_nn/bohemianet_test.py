"""Run a convolutional neural network on the test set
   and save the results to predictions.csv
"""
import numpy as np
import pandas as pd
from bohemianet import Bohemianet
from bohemianet_input import get_test_imgs


if __name__ == "__main__":
    save_path = "predictions.csv"

    bohemianet = Bohemianet(model_name="model1")
    bohemianet.load()

    batch_size = 64

    num_ids = 6600

    df = pd.DataFrame(index=range(num_ids), columns=['id', 'class'])
    df['id'] = np.arange(num_ids)

    i = 0
    for x_batch in get_test_imgs():
        classes = bohemianet.eval_classes(x_batch)
        df['class'][i:i+batch_size] = classes
        i += batch_size

    df.to_csv(save_path, index=False)
