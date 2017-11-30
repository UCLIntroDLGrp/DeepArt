# An example transfer learning script

from TransferLearning.VGG16 import VGG16
from keras.optimizers import SGD
from sklearn.metrics import f1_score, log_loss
from TransferLearning.transferLearning import refactorOutputs,setTrainableLayers,fineTune
from Preprocessing.preprocessing import generate_training_and_testing_data
from Utilities.utilities import selectData, collapseVectors

if __name__ == '__main__':
    #Get the data:
    crop_dims = (224, 224)
    directory = "../Art_Data_sm"
    number_of_crops = 1
    test_size = 1.0 / 3
    train_size = 2.0 / 3
    X_train, X_test, Y_train, Y_test = generate_training_and_testing_data(directory, crop_dims,
                                                                          number_of_crops, test_size, train_size)

    #Select only few training examples - uncomment for quick testing
    #X_train= selectData(X_train,50)
    #Y_train = selectData(y_train,50)
    #X_test = selectData(X_test,20)
    #Y_test = selectData(y_test,20)

    #Hyperparameters
    batch_size = 10
    nb_epoch = 1
    num_classes=8
    loss = 'categorical_crossentropy'
    metrics = ['accuracy']


    #Model on imagenet and optimizer instantiation
    model = VGG16(include_top=True, weights='imagenet')
    sgd = SGD(lr=1e-3, decay=1e-6, momentum=0.9, nesterov=True)


    #Do the transfer learning
    model = refactorOutputs(model,num_classes,True)
    model = setTrainableLayers(model,1)
    model = fineTune(model,batch_size,nb_epoch,sgd,loss,metrics,X_train,Y_train,X_test,Y_test,True)

    # Make predictions
    predictions_valid = model.predict(X_test, batch_size=batch_size, verbose=1)

    # Cross-entropy loss score
    log_loss = log_loss(Y_test, predictions_valid)
    collapsed_predictions = collapseVectors(predictions_valid)
    score = f1_score(Y_test, collapsed_predictions, average='macro')
    print(log_loss)
    print(score)