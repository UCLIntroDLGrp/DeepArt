import sys
import os
sys.path.insert(0, os.path.realpath('../'))
import numpy as np
from Utilities.utilities import selectData
from Preprocessing.preprocessing import generate_cropped_training_and_test_data,crop_data_from_load
from pickle import dump

from CustomNet.CustomNet import customModel, train

os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3"




sm_train_data = False
debug_data = False


#####################################
crop_dims = (224, 224)  ### AT MINIMUM must be (197,197)
input_shape = (224,224,3) #MUST MATCH CROP DIMS
number_of_crops = 4
dropout = True
####################################

if(sm_train_data):

    # Get the data:
    #directory = "../wikiart"
    directory = '../Art_Data_sm'
    validation_size = 10.0 / 100
    train_size = 80.0 / 100
    X_train, X_validation, Y_train, Y_validation = generate_cropped_training_and_test_data(directory,
                                                                               crop_dims,
                                                                               number_of_crops,
                                                                               validation_size,
                                                                               train_size,
                                                                               10)

else:
    if(not debug_data):
        X_train = np.load("../../../../../ml/2017/DeepArt/SavedData/X_train.npy")
        X_validation = np.load("../../../../../ml/2017/DeepArt/SavedData/X_validation.npy")
        Y_train = np.load("../../../../../ml/2017/DeepArt/SavedData/Y_train.npy")
        Y_validation = np.load("../../../../../ml/2017/DeepArt/SavedData/Y_validation.npy")

        X_train, X_validation, Y_train, Y_validation = crop_data_from_load(
            X_train, X_validation, Y_train, Y_validation, crop_dims, number_of_crops)

    else:
        X_train = np.load("../SavedData/X_train.npy")
        X_validation = np.load("../SavedData/X_validation.npy")
        Y_train = np.load("../SavedData/Y_train.npy")
        Y_validation = np.load("../SavedData/Y_validation.npy")
        #X_train, X_validation, Y_train, Y_validation = crop_data_from_load(
        #    X_train, X_validation, Y_train, Y_validation, crop_dims, number_of_crops)


if (debug_data):
    # Select only few training examples - uncomment for quick testing
    X_train = selectData(X_train, 16)
    Y_train = selectData(Y_train, 16)
    X_validation = selectData(X_validation, 8)
    Y_validation = selectData(Y_validation, 8)



##########################

if(not debug_data):
    batch_size = 32
    nb_epoch = 10
    num_classes = 7
else:
    batch_size = 4
    nb_epoch = 1
    num_classes = 8
############################


class Args(object):
    def __init__(self, batch_size, nb_epoch):  # lam_recon,
        self.batch_size = batch_size
        self.epochs = nb_epoch


args = Args(batch_size,nb_epoch)

model = customModel(input_shape,num_classes,dropout)

model.summary()

model , history = train(model=model, data=((X_train, Y_train), (X_validation, Y_validation)), args=args)

print("Finishing Training")

model.save("../SavedData/Experiment1CustomNet.h5")
model.save_weights("../SavedData/Experiment1CustomNetWeights.h5")
pickle_out = open("../SavedData/Experiment1CustomNetHistory.pickle", "wb")
dump(history.history, pickle_out)
pickle_out.close()
