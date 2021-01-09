from struct import unpack
from array import array
import numpy as np
import matplotlib.pyplot as plt
from math import e

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
    return (train_images, train_labels), (test_images, test_labels), (validate_images, validate_labels)


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
    numerator = 1
    denominator = 1 + e ** (-1 * x)
    return numerator/denominator

# ==================================================================================================================== #


def main():
    train_data, test_data = import_data()


if __name__ == '__main__':
    main()
