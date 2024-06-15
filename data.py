import numpy as np
import pathlib


def get_mnist():
    # record the mnist.npz file which is a numpy archive file of numpy arrays
    with np.load(f"{pathlib.Path(__file__).parent.absolute()}/data/mnist.npz") as f:
        images, labels = f["x_train"], f["y_train"]
        # normalize the images. this scales the images from the original range to a range 0-1
        images = images.astype("float32") / 255
        # the images are reshaped in a 2D array where every image is flattened into a single vector for example
        # a 28x28 image is a vector of lenght 784
        images = np.reshape(images, (images.shape[0], images.shape[1] * images.shape[2]))

        # one_hot encoding transform the 10 labels in a identity matrix. for example 3 becomes [0,0,0,1,0,0,0,0,0,0]
        labels = np.eye(10)[labels]

    return images, labels
