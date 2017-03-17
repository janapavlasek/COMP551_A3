"""Load the training data and prepare it for training the cnn

Attributes:
    IM_SIZE (int): input img length and width
    NUM_CHANNELS (int): input image color channels
    NUM_CLASSES (int): number of output classes
    RANDOM_KEY (int): random_key for train test split
    X_PATH (str): path to train images
    X_TEST_PATH (str): path to test images
    Y_PATH (str): path to labels
"""
import cv2
import time
import numpy as np
from sklearn.model_selection import train_test_split
import imgaug as ia
from imgaug import augmenters as iaa


X_PATH = "../data/tinyX.npy"
Y_PATH = "../data/tinyY.npy"
X_TEST_PATH = "../data/tinyX_test.npy"

IM_SIZE = 64
NUM_CHANNELS = 3
NUM_CLASSES = 40

RANDOM_KEY = 0


def get_train_batches(num_batches, batch_size=64, augment=True, split=True):
    """Generator for creating training batches of images and labels

    Args:
        num_batches (int): number of batches to generate
        batch_size (int, optional): number of images in each batch
        augment (bool, optional): whether to augment the images
        split (bool, optional): whether to create a validation set

    Yields:
        (img_array, label_array): batch of images and their labels
    """
    ims = np.load(X_PATH)
    labels = np.load(Y_PATH)

    if split:
        trainX, _, trainY, _ = train_test_split(
            ims, labels, train_size=0.9, random_state=RANDOM_KEY)
    else:
        trainX = ims
        trainY = labels

    for _ in range(num_batches):
        rand_inds = np.random.randint(trainX.shape[0], size=batch_size)

        batch_ims = trainX[rand_inds].transpose(0, 3, 2, 1)
        if augment:
            batch_ims = augment_batch(batch_ims)
        batch_ims = batch_ims.reshape(
            batch_size, IM_SIZE * IM_SIZE * NUM_CHANNELS)
        batch_ims = batch_ims / 255

        batch_labs = np.zeros((batch_size, NUM_CLASSES), dtype='int8')
        batch_labs[np.arange(batch_size), trainY[rand_inds]] = 1
        yield batch_ims, batch_labs


def get_valid_batches(batch_size=64):
    """Generator to yield the images and labels from the validation set

    Args:
        batch_size (int, optional): number of images in each batch

    Yields:
        (img_array, label_array): batch of images and their labels
    """
    ims = np.load(X_PATH)
    labels = np.load(Y_PATH)

    _, validX, _, validY = train_test_split(
        ims, labels, train_size=0.9, random_state=RANDOM_KEY)

    n = validX.shape[0]

    for i in range(0, n, batch_size):
        if n - i < batch_size:
            batch_size = n - i

        batch_ims = validX[i:i+batch_size].transpose(0, 3, 2, 1) / 255
        batch_ims = batch_ims.reshape(
            batch_size, IM_SIZE * IM_SIZE * NUM_CHANNELS)

        batch_labs = np.zeros((batch_size, NUM_CLASSES), dtype='int8')
        batch_labs[np.arange(batch_size), validY[i:i+batch_size]] = 1
        yield batch_ims, batch_labs


def get_test_imgs(batch_size=64):
    """Generator to yield the images from the test set

    Args:
        batch_size (int, optional): number of images in each batch

    Yields:
        img_array: batch of images
    """
    testX = np.load(X_TEST_PATH)
    n = testX.shape[0]

    for i in range(0, n, batch_size):
        if n - i < batch_size:
            batch_size = n - i
        batch_ims = testX[i:i+batch_size].transpose(0, 3, 2, 1) / 255
        yield batch_ims.reshape(batch_size, IM_SIZE * IM_SIZE * NUM_CHANNELS)


def augment_batch(batch):
    """apply transformations to an image batch

    Args:
        batch (img_array): batch of images to augmnet

    Returns:
        img_array: batch of augmented images
    """
    st = lambda aug: iaa.Sometimes(0.5, aug)

    seq = iaa.Sequential([
            iaa.Fliplr(0.5), # horizontally flip 50% of all images
            st(iaa.Crop(percent=(0, 0.1))), # crop images by 0-10% of their height/width
            st(iaa.GaussianBlur((0, 0.6))), # blur images with a sigma between 0 and 3.0
            st(iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.2), per_channel=0.5)), # add gaussian noise to images
            st(iaa.Dropout((0.0, 0.05), per_channel=0)), # randomly remove up to 10% of the pixels
            st(iaa.Add((-10, 10), per_channel=0.5)), # change brightness of images (by -10 to 10 of original value)
            st(iaa.Multiply((0.5, 1.5), per_channel=0.5)), # change brightness of images (50-150% of original value)
            st(iaa.ContrastNormalization((0.6, 2.0), per_channel=0)), # improve or worsen the contrast
            st(iaa.Affine(
                scale={"x": (0.8, 1.2), "y": (0.8, 1.2)}, # scale images to 80-120% of their size, individually per axis
                translate_px={"x": (-16, 16), "y": (-16, 16)}, # translate by -16 to +16 pixels (per axis)
                rotate=(-10, 10), # rotate by -45 to +45 degrees
                shear=(-10, 10), # shear by -16 to +16 degrees
                # order=ia.ALL, # use any of scikit-image's interpolation methods
                # cval=(0, 1.0) # if mode is constant, use a cval between 0 and 1.0
                # mode=ia.ALL # use any of scikit-image's warping modes (see 2nd image from the top for examples)
            )),
            st(iaa.ElasticTransformation(alpha=(0.1, 0.6), sigma=0.2)) # apply elastic transformations with random strengths
        ],
        random_order=True # do all of the above in random order
    )

    return seq.augment_images(batch)


if __name__ == '__main__':
    start = time.time()
    for ims, labs in get_train_batches(10, batch_size=64, augment=True):
        pass
    print(time.time() - start)
