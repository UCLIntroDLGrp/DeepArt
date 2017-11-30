import numpy as np
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

def load_cropped_images(directory_path, crop_dimensions, number_of_crops):
	'''
	Arguments:
		directory_path: the path for where image directories exist
		crop_dimensions: the dimensions of the cropping window
		number_of_crops: the number of crops to be produced from the image
	Returns:
		cropped_images: an array of cropped images
		cropped_labels: an array of corresponding genres
	'''
	images, labels = load_images(directory_path)
	cropped_images = []
	cropped_labels = []

	for image, label in zip(images, labels):
		cropped = generate_random_crops(image, crop_dimensions, number_of_crops)
		for crop in cropped:
			cropped_images.append(crop)
			cropped_labels.append(label)

	label_set = set(cropped_labels)

	cropped_labels = one_hot_encoding(cropped_labels)

	return np.array(cropped_images), np.array(cropped_labels).reshape(-1, len(label_set))

def generate_training_and_testing_data(directory_path, crop_dimensions, number_of_crops, test_size, train_size):
	'''
	Arguments:
		directory_path: the path for where image directories exist
		crop_dimensions: the dimensions of the cropping window
		number_of_crops: the number of crops to be produced from the image
		test_size: test set proportion
		train_size: train set proportion
	Returns:
		X_train: X training data
		X_test: X testing data
		y_train: y training data
		y_test: y testing data
	'''
	images, labels = load_cropped_images(directory_path, crop_dimensions, number_of_crops)

	X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=test_size, train_size=train_size)

	return X_train, X_test, y_train, y_test
