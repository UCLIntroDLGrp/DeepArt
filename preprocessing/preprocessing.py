import numpy as np
import matplotlib.image as img
import matplotlib.pyplot as plt
import os 
from sklearn.model_selection import  train_test_split
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
	resize_image = image.resize((width/scale, height/scale))
	return resize_image

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

def generate_training_and_test_data(directory_path, test_size, train_size):
	images, labels = load_images(directory_path)
	labels = one_hot_encoding(labels)

	X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=test_size, train_size=train_size)
	return X_train, X_test, y_train, y_test

def generate_cropped_training_and_test_data(directory_path, crop_dimensions, number_of_crops, test_size, train_size):
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
	X_train, X_test, y_train, y_test = generate_training_and_test_data(directory_path, test_size, train_size)
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
