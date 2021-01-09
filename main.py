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


def import_set(labels_path, labels_number, images_path, images_number):
    labels = load_labels(labels_path, labels_number)
    images = load_images(images_path, images_number)

    return pair_images_and_labels(images, labels)


def import_data():
    train_data = import_set(TRAIN_LABELS_PATH, 2049, TRAIN_IMAGES_PATH, 2051)
    test_data = import_set(TEST_LABELS_PATH, 2049, TEST_IMAGES_PATH, 2051)
    train_data, validation_data = split_data(train_data, VALIDATE_SET_LENGTH)

    train_data = convert_result_into_vector(train_data, 10)

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
            image = image.reshape(784, 1)
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


def convert_result_into_vector(data_set, vector_size):
    new_data_set = []

    for entry in data_set:
        vector = np.zeros((vector_size, 1))
        vector[entry[1]] = 1.0
        new_entry = (entry[0], vector)
        new_data_set.append(new_entry)
    return new_data_set


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



def calculate_network_result(argument, weights, biases):
    for weight, bias in zip(weights, biases):
        argument = np.dot(weight, argument) + bias
        argument = sigmoid(argument)
    return argument


def evaluate(test_data, weights, biases):
    correct_answers_count = 0.0
    for activations, correct_number in test_data:
        network_prediction = calculate_network_result(activations, weights, biases)
        network_prediction = np.argmax(network_prediction)
        if correct_number == network_prediction:
            correct_answers_count += 1.0
    correctness_coefficient = round(correct_answers_count / len(test_data) * 100, 2)
    return correctness_coefficient


# ==================================================================================================================== #


def main():
    train_data, test_data, validation_data = import_data()
    print("Data has been imported.")


if __name__ == '__main__':
    main()
