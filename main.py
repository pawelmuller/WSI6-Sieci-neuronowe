from struct import unpack
from array import array
import numpy as np
import matplotlib.pyplot as plt

TEST_IMAGES_PATH = 'Data/t10k-images-idx3-ubyte'
TEST_LABELS_PATH = 'Data/t10k-labels-idx1-ubyte'
TRAIN_IMAGES_PATH = 'Data/train-images-idx3-ubyte'
TRAIN_LABELS_PATH = 'Data/train-labels-idx1-ubyte'

IMAGES_OFFSET = 16
LABELS_OFFSET = 8


def import_data():
    train_labels = load_labels(TRAIN_LABELS_PATH, 2049)
    train_images = load_images(TRAIN_IMAGES_PATH, 2051)
    test_labels = load_labels(TEST_LABELS_PATH, 2049)
    test_images = load_images(TEST_IMAGES_PATH, 2051)
    return (train_images, train_labels), (test_images, test_labels)


def load_images(filename, correct_magic_number):
    with open(filename, 'rb') as file:
        magic_number, number_of_images, image_height, image_width = unpack(">IIII", file.read(IMAGES_OFFSET))
        if magic_number != correct_magic_number:
            raise ValueError(f'Incorrect magic number. Expected {correct_magic_number}, got {magic_number}.')
        data = array("B", file.read())

        images = []
        for i in range(number_of_images):
            image = np.array(data[i * image_height * image_width:(i + 1) * image_height * image_width])
            image = image.reshape(image_height, image_width)
            images.append(image)
    return images


def load_labels(filename, correct_magic_number):
    with open(filename, 'rb') as file:
        magic_number, number_of_items = unpack(">II", file.read(LABELS_OFFSET))
        if magic_number != correct_magic_number:
            raise ValueError(f'Incorrect magic number. Expected {correct_magic_number}, got {magic_number}.')
        labels = array("B", file.read())
    return labels


def draw_image(image, label):
    plt.imshow(image, cmap=plt.cm.gray)
    plt.title(label)
    plt.show()


def main():
    train_data, test_data = import_data()
    for i in range(0, 50, 5):
        draw_image(train_data[0][i], str(train_data[1][i]))


if __name__ == '__main__':
    main()
