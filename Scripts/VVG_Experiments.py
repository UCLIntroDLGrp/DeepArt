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

sys.path.insert(0, os.path.realpath('../'))

from keras.optimizers import Adam
from sklearn.metrics import f1_score, log_loss
from TransferLearning.transferLearning import refactorOutputs, setTrainableLayers, fineTune
from Preprocessing.preprocessing import generate_cropped_training_and_test_data
from Utilities.utilities import selectData, collapseVectors
from keras.applications.vgg16 import VGG16

if __name__ == '__main__':
    # Get the data:
    crop_dims = (224, 224)
    directory = "../Art_Data_sm"
    number_of_crops = 4
    validation_size = 10.0 / 100
    train_size = 80.0 / 100
    X_train, X_validation, Y_train, Y_validation = generate_cropped_training_and_test_data(directory,
                                                                               crop_dims,
                                                                               number_of_crops,
                                                                               validation_size,
                                                                               train_size,
                                                                               10)

    # Select only few training examples - uncomment for quick testing
    X_train = selectData(X_train, 30)
    Y_train = selectData(Y_train, 30)
    X_validation = selectData(X_validation, 10)
    Y_validation = selectData(Y_validation, 10)

    # Hyperparameters
    batch_size = 5
    nb_epoch = 2
    patience = 10
    num_classes = 8
    loss = 'categorical_crossentropy'
    metrics = ['accuracy']

    # Model on imagenet and optimizer instantiation
    model = VGG16(include_top=True, weights='imagenet')
    model.summary()
    opt = Adam()

    # Do the transfer learning
    model = refactorOutputs(model, num_classes, True)
    model = fineTune(model, batch_size, nb_epoch, opt, loss, metrics, patience, X_train, Y_train, X_validation,
                     Y_validation,
                     "Experiment1History.", False)

    model.save("Experiment1Model.h5")
