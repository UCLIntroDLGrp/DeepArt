
import sys
import os
import numpy as np
from Preprocessing.preprocessing import generate_training_and_test_data

sys.path.insert(0, os.path.realpath('../'))

if __name__ == '__main__':
    # Get the data:
    crop_dims = (224, 224)
    directory = '../Art_Data_sm'
    number_of_crops = 4
    validation_size = 10.0 / 100
    train_size = 80.0 / 100
    X_train, X_validation, Y_train, Y_validation = generate_training_and_test_data(directory,
                                                                                   validation_size,
                                                                                   train_size,
                                                                                   image_selection_percent=10)
    print("Saving...")

    np.save("../SavedData/X_train", X_train)
    np.save("../SavedData/X_validation", X_validation)
    np.save("../SavedData/Y_train", Y_train)
    np.save("../SavedData/Y_validation", Y_validation)
