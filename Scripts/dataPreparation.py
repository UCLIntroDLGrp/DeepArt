
import sys
import os
import numpy as np

sys.path.insert(0, os.path.realpath('../'))
from Preprocessing.preprocessing import generate_training_and_test_data

if __name__ == '__main__':
    # Get the data:
    crop_dims = (224, 224)
    directory = '../Art_Data_sm'
    number_of_crops = 1
    validation_size = 10.0 / 100
    train_size = 80.0 / 100
    X_test, Y_test = generate_training_and_test_data(directory,
                                                     validation_size,
                                                     train_size,
                                                     image_selection_percent=10,
                                                     test_set=True)
    print("Saving...")

    np.save("../../../../../ml/2017/DeepArt/SavedData/X_test.npy", X_test)
    np.save("../../../../../ml/2017/DeepArt/SavedData/Y_test.npy", Y_test)
