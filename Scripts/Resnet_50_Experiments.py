# Load the images with crop 224,224
# split with train/validation/test
# Load VGG with imageNet weights and 8 classes

# FUNCTION  evalutate
# Save model history
# Save the model

# Experiemnt1
# Retrain all layers
# Run evaluate

# Experiment2
# Freeze first 3 blocks of VGG
# Run evaluate

# Experiment3
# Free everything but last layer
# Run evaluate

# An example transfer learning script
import sys
import os

import numpy as np

sys.path.insert(0, os.path.realpath('../'))

from keras.optimizers import Adam

from TransferLearning.transferLearning import refactorOutputs, setTrainableLayers,freezeLayersUpTo, fineTune
from Preprocessing.preprocessing import generate_cropped_training_and_test_data
from Utilities.utilities import selectData
from keras.applications.resnet50 import ResNet50

if __name__ == '__main__':
    sm_train_data = False
    debug_data = True

    if(sm_train_data):

        # Get the data:
        crop_dims = (224, 224)
        #directory = "../wikiart"
        directory = '../Art_Data_sm'
        number_of_crops = 4
        validation_size = 10.0 / 100
        train_size = 80.0 / 100
        X_train, X_validation, Y_train, Y_validation = generate_cropped_training_and_test_data(directory,
                                                                                   crop_dims,
                                                                                   number_of_crops,
                                                                                   validation_size,
                                                                                   train_size,
                                                                                   10)

    else:
        X_train = np.load("../SavedData/X_train.npy")
        X_validation = np.load("../SavedData/X_validation.npy")
        Y_train = np.load("../SavedData/Y_train.npy")
        Y_validation = np.load("../SavedData/Y_validation.npy")


    if (debug_data):
        # Select only few training examples - uncomment for quick testing
        X_train = selectData(X_train, 32)
        Y_train = selectData(Y_train, 32)
        X_validation = selectData(X_validation, 16)
        Y_validation = selectData(Y_validation, 16)


    # Hyperparameters
    batch_size = 8
    nb_epoch = 2
    patience = 1
    num_classes = 8
    loss = 'categorical_crossentropy'
    metrics = ['accuracy']



####### Experiment 1


    # Model on imagenet and optimizer instantiation
    model = ResNet50(include_top=True, weights='imagenet')
    opt = Adam()
    # Do the transfer learning
    model = refactorOutputs(model, num_classes, True)

    #for layer in model.layers:
     #   layer.name = "Experiment_1_" + layer.name

    model = fineTune(model, batch_size, nb_epoch, opt, loss, metrics, patience, X_train, Y_train, X_validation,
                     Y_validation,
                     "../SavedData/Experiment1HistoryResnet.", False)

    model.save("../SavedData/Experiment1Resnet.h5")
    ###########





####### Experiment 2


    # Model on imagenet and optimizer instantiation
    model = ResNet50(include_top=True, weights='imagenet')
    opt = Adam()

    # Do the transfer learning
    model = refactorOutputs(model, num_classes, True)

    #for layer in model.layers:
     #   layer.name = "Experiment_2_" + layer.name

    model = freezeLayersUpTo(model, "Experiment_2_activation_13")
    model = fineTune(model, batch_size, nb_epoch, opt, loss, metrics, patience, X_train, Y_train, X_validation,
                     Y_validation,
                     "../SavedData/Experiment2HistoryResnet.", False)

    model.save("../SavedData/Experiment2Resnet.h5")

###########


####### Experiment 3


    # Model on imagenet and optimizer instantiation
    model = ResNet50(include_top=True, weights='imagenet')
    opt = Adam()

    # Do the transfer learning
    model = refactorOutputs(model, num_classes, True)
   # for layer in model.layers:
    #    layer.name = "Experiment_3_" + layer.name
    model = setTrainableLayers(model, 1)
    model = fineTune(model, batch_size, nb_epoch, opt, loss, metrics, patience, X_train, Y_train, X_validation,
                     Y_validation,
                     "../SavedData/Experiment3HistoryResnet.", False)

    model.save("../SavedData/Experiment3Resnet.h5")
###########


