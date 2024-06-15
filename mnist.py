from data import get_mnist
import numpy as np
import matplotlib.pyplot as plt

images, labels = get_mnist()

# print("images: ")
# print(images)

# print("labels: ")
# print(labels)

"""
w = weights, b = bias, i = input, h = hidden, o = output, l = label
w_i_h = weights from input layer to hidden layer
w_h_o = weight from hidden layer to output layer
b_i_h = bias from input to hidden
b_h_o = bias from hidden to output
"""

# generate random numbers with range from -0.5 to 0.5
# (20,784) is for specify the form of the resulting array which is a 2D matrix of 20 rows and 784 columns
w_i_h = np.random.uniform(-0.5, 0.5, (20, 784))
w_h_o = np.random.uniform(-0.5, 0.5, (10, 20))
b_i_h = np.zeros((20, 1))
b_h_o = np.zeros((10, 1))
learn_rate = 0.01
nr_correct = 0
epochs = 3

for epochs in range(epochs):
    for img, lbs in zip(images, labels):
        img.shape += (1,)
        lbs.shape += (1,)
        # Forward propagation input -> hidden
        # @ matrix multiplication
        h_pre = b_i_h + w_i_h @ img
        h = 1 / (1 + np.exp(-h_pre))
        # Forward propagation hidden -> output
        o_pre = b_h_o + w_h_o @ h
        o = 1 / (1 + np.exp(-o_pre))

        # cost/ error calculation

        e = 1 / len(o) * np.sum((o - lbs) ** 2, axis=0)
        nr_correct += int(np.argmax(o) == np.argmax(lbs))

        # back propagation output -> hidden (cost function derivative)
        delta_o = o - lbs
        w_h_o += -learn_rate * delta_o @ np.transpose(h)
        b_h_o += -learn_rate * delta_o

        # back propagation output hidden -> output activation function derivative
        delta_h = np.transpose(w_h_o) @ delta_o * (h * (1 - h))
        w_i_h += -learn_rate * delta_h @ np.transpose(img)
        b_i_h += -learn_rate * delta_h

    # show accuracy for this epoch
    print(nr_correct)
    print(f"acc: {round((nr_correct / images.shape[0]) * 100, 2)}%")
    nr_correct = 0

while True:
    index = int(input("enter a number ( 0 - 59999: "))
    img = images[index]
    plt.imshow(img.reshape(28, 28), cmap="Greys")

    img.shape += (1,)
    # forward propagation input -> hidden
    h_pre = b_i_h + w_i_h @ img.reshape(784, 1)
    h = 1 / (1 + np.exp(-h_pre))
    # forward propagation hidden -> output
    o_pre = b_h_o + w_h_o @ h
    o = 1 / (1 + np.exp(-o_pre))

    plt.title(f"test: {o.argmax()}")
    plt.show()
