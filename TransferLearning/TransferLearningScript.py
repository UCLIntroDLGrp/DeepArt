from TransferLearning.VGG16 import VGG16
from keras.optimizers import SGD
from sklearn.metrics import log_loss
from TransferLearning.transferLearningUtilities import refactorOutputs,setTrainableLayers,fineTune
from TransferLearning.load_cifar10 import load_cifar10_data

if __name__ == '__main__':

    #Hyperparameters
    batch_size = 10
    nb_epoch = 1
    num_classes=10
    loss = 'categorical_crossentropy'
    metrics = ['accuracy']


    #Model on imagenet and optimizer instantiation
    model = VGG16(include_top=True, weights='imagenet')
    sgd = SGD(lr=1e-3, decay=1e-6, momentum=0.9, nesterov=True)

    #Fetch the data
    img_rows, img_cols = 224, 224
    X_train, Y_train, X_valid, Y_valid = load_cifar10_data(img_rows, img_cols)
    X_train_new = X_train[:100, :, :, :]
    Y_train_new = Y_train[:100, :]
    X_valid_new = X_valid[:30, :, :, :]
    Y_valid_new = Y_valid[:30, :]

    #Do the transfer learning
    model = refactorOutputs(model,num_classes,True)
    model = setTrainableLayers(model,1)
    model = fineTune(model,batch_size,nb_epoch,sgd,loss,metrics,X_train_new,Y_train_new,X_valid_new,Y_valid_new,True)

    # Make predictions
    predictions_valid = model.predict(X_valid_new, batch_size=batch_size, verbose=1)

    # Cross-entropy loss score
    score = log_loss(Y_valid_new, predictions_valid)
    print(score)
