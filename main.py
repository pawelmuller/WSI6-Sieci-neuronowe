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
    """
    Loads and pairs data.
    :param labels_path: Path to labels file.
    :param labels_number: Magic number for labels validation.
    :param images_path: Path to images file.
    :param images_number: Magic number for images validation.
    :return: List of (image, label) for each entry.
    """
    labels = load_labels(labels_path, labels_number)
    images = load_images(images_path, images_number)

    return pair_images_and_labels(images, labels)


def import_data():
    """
    Imports and arranges train, test and validation data.
    :return: Ready to use data sets.
    """
    # Importing data sets
    train_data = import_set(TRAIN_LABELS_PATH, 2049, TRAIN_IMAGES_PATH, 2051)
    test_data = import_set(TEST_LABELS_PATH, 2049, TEST_IMAGES_PATH, 2051)

    # Dividing train set into new train set and validation set
    train_data, validation_data = split_data(train_data, VALIDATE_SET_LENGTH)

    # Converting train data labels into one-hot array
    train_data = convert_result_into_vector(train_data, 10)

    return train_data, test_data, validation_data


def split_data(data, set_length):
    """
    Splits given list/array into two.
    :param data: Input data set.
    :param set_length: How big should one of the sets be (the other will contain other elements).
    :return: Divided data sets.
    """
    first_set = data[:-set_length]
    second_set = data[-set_length:]
    return first_set, second_set


def load_images(filename, correct_magic_number):
    """
    Reads image file and converts data into numpy arrays.
    :param filename: File name or path to the file.
    :param correct_magic_number: Validation number.
    :return: Array of imported images.
    """
    with open(filename, 'rb') as file:
        # Reading file header
        magic_number, number_of_images, image_height, image_width = unpack(">IIII", file.read(IMAGES_OFFSET))

        # Data validation
        if magic_number != correct_magic_number:
            raise ValueError(f'Incorrect magic number. Expected {correct_magic_number}, got {magic_number}.')

        # Initialising unsigned char array
        data = array("B", file.read())

        # Loading images
        images = []
        for i in range(number_of_images):
            left_index = i * image_height * image_width
            right_index = (i + 1) * image_height * image_width
            image = np.array(data[left_index:right_index])
            # Reshaping and scaling images
            image = image.reshape(784, 1)
            image = map_values(image, 0.01, 1)
            images.append(image)
    return images


def load_labels(filename, correct_magic_number):
    """
    Reads label file.
    :param filename: File name or path to the file.
    :param correct_magic_number: Validation number.
    :return: Array of imported labels.
    """
    with open(filename, 'rb') as file:
        # Reading file header
        magic_number, number_of_items = unpack(">II", file.read(LABELS_OFFSET))

        # Data validation
        if magic_number != correct_magic_number:
            raise ValueError(f'Incorrect magic number. Expected {correct_magic_number}, got {magic_number}.')

        # Initialising unsigned char array
        labels = array("B", file.read())
    return labels


def pair_images_and_labels(images, labels):
    """
    Pairs each image and label in tuple.
    :param images: Images array.
    :param labels: Labels array.
    :return: List of (image, label) tuples.
    """
    return [pair for pair in zip(images, labels)]


def convert_result_into_vector(data_set, vector_size):
    """
    Converts each label for data set into one-hot vector.
    :param data_set: Input data set.
    :param vector_size: Output vector size.
    :return: List of (image, one-hot vector label) tuples.
    """
    new_data_set = []

    for entry in data_set:
        # Creating vector filled with zeros.
        vector = np.zeros((vector_size, 1))
        # Changing respective vector element
        vector[entry[1]] = 1.0
        # Creating and appending new entry.
        new_entry = (entry[0], vector)
        new_data_set.append(new_entry)
    return new_data_set


def map_values(values, left, right):
    """
    Scales values in <left, right) half-open interval.
    :param values: Input values.
    :param left: Left bound (closed).
    :param right: Right bound (opened).
    :return: Scaled values.
    """
    fraction = (right - left) / max(values)
    values = values * fraction + left
    return values


def draw_image(image, label):
    """
    Draws and shows given image.
    :param image: Image values array.
    :param label: Title array.
    """
    image = image.reshape(28, 28)
    plt.imshow(image, cmap=plt.cm.gray)
    plt.title(label)
    plt.show()


def show_some_images(data):
    for i in range(0, 50, 5):
        draw_image(data[0][i], str(data[1][i]))


# ==================================================================================================================== #


def sigmoid(x):
    """
    Calculates sigmoid function.
    :param x: Argument.
    :return: Value.
    """
    numerator = 1.0
    denominator = 1.0 + e ** (-1.0 * x)
    return numerator/denominator


def sigmoid_derivative(x):
    """
    Calculates sigmoid derivative function.
    :param x: Argument.
    :return: Value.
    """
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
        # Looking for highest number in result array (it's index will equal network's number guess)
        network_prediction = calculate_network_result(activations, weights, biases)
        network_prediction = np.argmax(network_prediction)
        if correct_number == network_prediction:
            correct_answers_count += 1.0
    correctness_coefficient = round(correct_answers_count / len(test_data) * 100, 2)
    return correctness_coefficient


def learn_network(train_data, epochs_count, subset_size, step_size, layers_count, weights, biases, test_data=None):
    for epoch in range(epochs_count):
        shuffle(train_data)

        # Dividing train set into smaller subsets
        subset_count = len(train_data) // subset_size
        subsets = np.array_split(train_data, subset_count)

        # 'Learning'
        for subset in subsets:
            weights, biases = calculate_new_parameters(subset, step_size, layers_count, weights, biases)

        # Evaluating each epoch if test set provided
        if test_data:
            print(f"Epoch {epoch:>2} correctness coefficient: {evaluate(test_data, weights, biases)}%")

    return weights, biases


def calculate_new_parameters(subset, step_size, layers_count, weights, biases):
    subset_size = len(subset)

    # v - ∇
    v_weights = prepare_v_parameter_list(weights)
    v_biases = prepare_v_parameter_list(biases)

    for activations, correct_number in subset:
        delta_v_weights, delta_v_biases = backpropagation(activations, correct_number, layers_count, weights, biases)
        v_biases = [dnb + nb for dnb, nb in zip(delta_v_biases, v_biases)]
        v_weights = [dnw + nw for dnw, nw in zip(delta_v_weights, v_weights)]

    sizes_quotient = step_size / subset_size
    new_weights = get_new_parameter(weights, v_weights, sizes_quotient)
    new_biases = get_new_parameter(biases, v_biases, sizes_quotient)

    return new_weights, new_biases


def prepare_v_parameter_list(parameter):
    parameter_list = []
    for element in parameter:
        parameter_list.append(np.zeros(element.shape))
    return parameter_list


def get_new_parameter(parameter, v_parameter, sizes_quotient):
    new_parameter = []
    for p, np in zip(parameter, v_parameter):
        new_parameter.append(p - sizes_quotient * np)
    return new_parameter


# ==================================================================================================================== #


def main():
    train_data, test_data, validation_data = import_data()
    print("Data has been imported.")


if __name__ == '__main__':
    main()
