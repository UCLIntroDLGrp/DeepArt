import numpy as np
import matplotlib.pyplot as plt

def crop_image(image, positions, crop_dimensions):
	'''
	Arguments: 
		image: the image to be cropped
		positions: the x and y positions of the top left cropping window
		crop_dimensions: the dimensions of the cropping window
	Returns:
		cropped_image: the newly cropped image
	'''
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