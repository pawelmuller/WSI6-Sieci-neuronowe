from struct import unpack
from array import array
import numpy as np
import matplotlib.pyplot as plt
from math import e
from numpy.random import randn
from random import shuffle

# Global settings.
TEST_IMAGES_PATH = 'Data/t10k-images-idx3-ubyte'
TEST_LABELS_PATH = 'Data/t10k-labels-idx1-ubyte'
TRAIN_IMAGES_PATH = 'Data/train-images-idx3-ubyte'
TRAIN_LABELS_PATH = 'Data/train-labels-idx1-ubyte'

IMAGES_OFFSET = 16
LABELS_OFFSET = 8

VALIDATE_SET_LENGTH = 10000


def import_data():
    train_labels = load_labels(TRAIN_LABELS_PATH, 2049)
    train_images = load_images(TRAIN_IMAGES_PATH, 2051)
    test_labels = load_labels(TEST_LABELS_PATH, 2049)
    test_images = load_images(TEST_IMAGES_PATH, 2051)
    train_labels, validate_labels = split_data(train_labels, VALIDATE_SET_LENGTH)
    train_images, validate_images = split_data(train_images, VALIDATE_SET_LENGTH)

    train_data = pair_images_and_labels(train_images, train_labels)
    test_data = pair_images_and_labels(test_images, test_labels)
    validation_data = pair_images_and_labels(validate_images, validate_labels)

    return train_data, test_data, validation_data


def split_data(data, set_length):
    first_set = data[:-set_length]
    second_set = data[-set_length:]
    return first_set, second_set


def load_images(filename, correct_magic_number):
    with open(filename, 'rb') as file:
        magic_number, number_of_images, image_height, image_width = unpack(">IIII", file.read(IMAGES_OFFSET))
        if magic_number != correct_magic_number:
            raise ValueError(f'Incorrect magic number. Expected {correct_magic_number}, got {magic_number}.')
        data = array("B", file.read())

        images = []
        for i in range(number_of_images):
            image = np.array(data[i * image_height * image_width:(i + 1) * image_height * image_width])
            # image = image.reshape(image_height, image_width)
            image = map_values(image, 0.01, 1)
            images.append(image)
    return images


def load_labels(filename, correct_magic_number):
    with open(filename, 'rb') as file:
        magic_number, number_of_items = unpack(">II", file.read(LABELS_OFFSET))
        if magic_number != correct_magic_number:
            raise ValueError(f'Incorrect magic number. Expected {correct_magic_number}, got {magic_number}.')
        labels = array("B", file.read())
    return labels


def pair_images_and_labels(images, labels):
    return [pair for pair in zip(images, labels)]


def map_values(values, left, right):
    fraction = (right - left) / max(values)
    values = values * fraction + left
    return values


def draw_image(image, label):
    image = image.reshape(28, 28)
    plt.imshow(image, cmap=plt.cm.gray)
    plt.title(label)
    plt.show()


def show_some_images(data):
    for i in range(0, 50, 5):
        draw_image(data[0][i], str(data[1][i]))


# ==================================================================================================================== #

def sigmoid(x):
    numerator = 1.0
    denominator = 1.0 + e ** (-1.0 * x)
    return numerator/denominator


def sigmoid_derivative(x):
    numerator = e ** (-1.0 * x)
    denominator = (e ** (-1.0 * x) + 1) ** 2
    return numerator/denominator

# ==================================================================================================================== #


def initialise_network(layer_sizes):
    layers_count = len(layer_sizes)
    biases = [np.random.randn(y, 1) for y in layer_sizes[1:]]
    weights = [np.random.randn(y, x) for x, y in zip(layer_sizes[:-1], layer_sizes[1:])]


def calculate_network_result(network_input, weights, biases):
    for weight, bias in zip(weights, biases):
        argument = np.dot(weight, network_input) + bias
        result = sigmoid(argument)
    return result


# ==================================================================================================================== #


def main():
    train_data, test_data, validation_data = import_data()
    print("Data has been imported.")


if __name__ == '__main__':
    main()
