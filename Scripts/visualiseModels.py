# An example transfer learning script
import sys
import os
sys.path.insert(0, os.path.realpath('../'))

from keras.optimizers import Adam
from sklearn.metrics import f1_score, log_loss
from TransferLearning.transferLearning import refactorOutputs,setTrainableLayers,fineTune
from Preprocessing.preprocessing import generate_cropped_training_and_test_data
from Utilities.utilities import selectData, collapseVectors
from vis.visualization import visualize_saliency, visualize_activation
from PIL import Image
from vis.utils import utils
from keras.applications.vgg16 import VGG16

if __name__ == '__main__':
    #Get the data:
    crop_dims = (224, 224)
    directory = "../Art_Data_sm"
    number_of_crops = 10
    test_size = 1.0 / 3
    train_size = 2.0 / 3
    X_train, X_test, Y_train, Y_test = generate_cropped_training_and_test_data(directory, crop_dims,
                                                                          number_of_crops, test_size, train_size)

    #Select only few training examples - uncomment for quick testing
    X_train= selectData(X_train,30)
    Y_train = selectData(Y_train,30)
    X_test = selectData(X_test,10)
    Y_test = selectData(Y_test,10)

    #Hyperparameters
    batch_size = 10
    nb_epoch = 2
    num_classes=8
    loss = 'categorical_crossentropy'
    metrics = ['accuracy']


    #Model on imagenet and optimizer instantiation
    model = VGG16(include_top=True, weights='imagenet')
    model.summary()
    opt = Adam()

    layer_index = utils.find_layer_idx(model,'block5_conv3')
    print(layer_index)
    ret = visualize_activation(model,layer_index, None, X_test[1,:,:,:]);

    img = Image.fromarray(ret, 'RGB')
    img.show()

    ret2 = visualize_saliency(model, layer_index, None, X_test[1, :, :, :]);

    img2 = Image.fromarray(ret2, 'RGB')
    img2.show()


    #Do the transfer learning
    model = refactorOutputs(model,num_classes,True)
    model = setTrainableLayers(model,3)
    model = fineTune(model,batch_size,nb_epoch,opt,loss,metrics,X_train,Y_train,X_test,Y_test,True)

    # Make predictions
    predictions_valid = model.predict(X_test, batch_size=batch_size, verbose=1)

    # Cross-entropy loss score
    log_loss = log_loss(Y_test, predictions_valid)
    collapsed_predictions = collapseVectors(predictions_valid)
    score = f1_score(Y_test, collapsed_predictions, average='macro')


    print(log_loss)
    print(score)