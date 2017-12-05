import preprocessing as preprocess
import os 
import shutil
import numpy as np 

original_image_directory = '../Test_directory_move'#../Art_Data_sm/'

class_dictionary = preprocess.read_directory(original_image_directory)
class_names, images = class_dictionary.keys(), class_dictionary.values()

new_image_directory = '../Test_data/'
os.makedirs(new_image_directory)

for name, image_list in zip(class_names, images):

	indices = np.arange(len(image_list))
	
	random_image_indices = np.random.choice(indices, int(0.1 * len(image_list)), replace=False)
	
	image_subset = [image_list[i] for i in random_image_indices]

	os.makedirs(new_image_directory + "/" + name)

	for image in image_subset:
		shutil.move(original_image_directory + "/" + name  + "/" + str(image), new_image_directory + "/" + name + "/" + str(image))