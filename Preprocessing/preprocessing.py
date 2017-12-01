import numpy as np
import os
from sklearn.model_selection import train_test_split
from skimage.transform import resize
from PIL import Image


def crop_image(image, positions, crop_dimensions):
    '''
    Arguments:
        image: the image to be cropped
        positions: the x and y positions of the top left cropping window
        crop_dimensions: the dimensions of the cropping window
    Returns:
        cropped_image: the newly cropped image
    '''
    image = np.array(image)
    height, width = crop_dimensions
    height_pos, width_pos = positions

    cropped_image = image[height_pos:height_pos + height, width_pos:width_pos + width]
    return cropped_image


def generate_random_crops(image, crop_dimensions, number_of_images):
    '''
    Arguments:
        image: the image to be cropped
        crop_dimensions: the dimensions of the cropping window
        number_of_images: the number of crops to be produced from the image
    Returns:
        cropped_images: an array of cropped images from image.
    '''
    image = np.array(image)
    image_height, image_width, _ = image.shape
    crop_height, crop_width = crop_dimensions

    cropped_images = []

    for i in range(number_of_images):
        random_height_position = np.random.randint(0, image_height - crop_height)
        random_width_position = np.random.randint(0, image_width - crop_width)

        cropped_image = crop_image(image,
                                   (random_height_position, random_width_position),
                                   crop_dimensions)

        cropped_images.append(cropped_image)

    return cropped_images


def resize_image(image, scale):
    '''
    Arguments:
        image: the image to be resized
        scale: resizing scale factor
    Returns:
        resized image: a scaled version of the original image
    '''
    height, width, _ = np.array(image).shape
    resize_image = image.resize((width / scale, height / scale))
    return resize_image


def resize_to_minimum(image, minimum_dimensions):
    return resize(image, minimum_dimensions)

def read_directory(directory_path):
    '''
    Arguments:
        directory_path: the path for where image directories exist
    Returns:
        class_dictionary: a dictionary with genres as keys and an array
                          of image names as values.
    '''
    get_dir_names = os.listdir(directory_path)

    class_dictionary = dict()

    for directory in get_dir_names:
        class_dictionary[directory] = os.listdir(directory_path + "/" + directory)

    return class_dictionary


def load_images(directory_path):
    '''
    Arguments:
        directory_path: the path for where image directories exist
    Returns:
        images: an array of all the art images
        label: the genre corresponding to each image
    '''
    class_dictionary = read_directory(directory_path)
    images = []
    labels = []

    for key, value in zip(class_dictionary.keys(), class_dictionary.values()):
        for image_name in value:
            image = Image.open(directory_path + "/" + key + "/" + image_name)

            images.append(np.array(image))
            labels.append(key)

    return images, labels


def select_images_of_similar_size(directory, percent, print_me=False):
    """
   Arguments
       percent: default 0.2 will remove the smallest and largest 20% of images,
       both in height and width

      Returns
       images_out: new set of images with smallest / largest percent removed
       num_before: number of images originally loaded
       num_after: number of images after removing smallest / largest percent
   """
    images_in, labels_in = load_images(directory)
    num_before = len(images_in)

    widths = []
    heights = []
    for image in images_in:
        widths.append(image.shape[0])
        heights.append(image.shape[1])

    if print_me == True:
        print("Max width before: ", np.max(widths))
        print("Min width before: ", np.min(widths))
        print("Max height before: ", np.max(heights))
        print("Min height before: ", np.min(heights))

    i_vanish = []
    index = 0
    for image in images_in:
        if image.shape[0] < np.percentile(widths, percent):
            i_vanish.append(index)
        elif image.shape[0] > np.percentile(widths, 100 - percent):
            i_vanish.append(index)
        elif image.shape[1] < np.percentile(heights, percent):
            i_vanish.append(index)
        elif image.shape[1] > np.percentile(heights, 100 - percent):
            i_vanish.append(index)
        index += 1

    if print_me == True:
        print("removing indices: ", i_vanish)

    for x in i_vanish[::-1]:
        del images_in[x]
        del labels_in[x]

    images_out = images_in
    labels_out = labels_in

    num_after = len(images_out)

    widths = []
    heights = []
    for image in images_out:
        widths.append(image.shape[0])
        heights.append(image.shape[1])

    if print_me == True:
        print("Max width after: ", np.max(widths))
        print("Min width after: ", np.min(widths))
        print("Max height after: ", np.max(heights))
        print("Min height after: ", np.min(heights))

    return images_out, labels_out

def generate_images_of_same_size(directory, percent):
    images, labels = select_images_of_similar_size(directory, 10)
    
    smallest_height = sorted(list(map(lambda x: x.shape, images)), key=lambda z: z[0])[0][0]
    smallest_width = sorted(list(map(lambda x: x.shape, images)), key=lambda z: z[1])[0][1]
    
    smallest_dimension = (smallest_height, smallest_width)
    
    scaled_images = []
    for image in images:
        scaled_images.append(resize_to_minimum(image, smallest_dimension))

    return scaled_images, labels

def one_hot_encoding(labels):
    '''
    Arguments:
        labels: an array of all the labels
    Returns:
        one_hot_encoded_labels: an array of all the one hot encoded labels
    '''
    labels_set = list(set(labels))
    one_hot_encoded_labels = []
    for label in labels:
        label_index = labels_set.index(label)

        vec = np.zeros((len(labels_set), 1))
        vec[label_index, 0] = 1
        one_hot_encoded_labels.append(vec)

    return one_hot_encoded_labels


def generate_training_and_test_data(directory_path, test_size, train_size, image_selection_percent, same_size=False):
    if(same_size):
        images, labels = generate_images_of_same_size(directory_path, percent=image_selection_percent)
    else:
        images, labels = select_images_of_similar_size(directory_path, percent=image_selection_percent)

    labels = np.array(one_hot_encoding(labels)).reshape(-1, 8)

    X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=test_size, train_size=train_size)
    return np.array(X_train), np.array(X_test), y_train, y_test

def generate_cropped_training_and_test_data(directory_path, crop_dimensions, number_of_crops, test_size, train_size,
                                            image_selection_percent=10):
    '''
    Arguments:
        directory_path: the path for where image directories exist
        crop_dimensions: the dimensions of the cropping window
        number_of_crops: the number of crops to be produced from the image
        test_size: testing proportion
        train_size: training proportion
    Returns:
        X_train, X_test, y_train, y_test (cropped)
    '''
    X_train, X_test, y_train, y_test = generate_training_and_test_data(directory_path, test_size, train_size,
                                                                       image_selection_percent)
    cropped_images_train = []
    cropped_labels_train = []
    cropped_images_test = []
    cropped_labels_test = []

    for image, label in zip(X_train, y_train):
        cropped = generate_random_crops(image, crop_dimensions, number_of_crops)
        for crop in cropped:
            cropped_images_train.append(crop)
            cropped_labels_train.append(label)

    for image, label in zip(X_test, y_test):
        cropped = generate_random_crops(image, crop_dimensions, number_of_crops)
        for crop in cropped:
            cropped_images_test.append(crop)
            cropped_labels_test.append(label)

    return np.array(cropped_images_train), np.array(cropped_images_test), np.array(cropped_labels_train).reshape(-1, 8), np.array(cropped_labels_test).reshape(-1, 8)
