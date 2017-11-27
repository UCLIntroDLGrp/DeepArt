from keras.layers import Dense
from keras.models import Model
from TransferLearning.VGG16 import VGG16

def transferLearn(model,layers_to_train,batch_size, nb_epoch,optimizer,loss,
                  metrics, X_train, Y_train, X_valid, Y_valid, outputs=None ):

    model.summary()
    if(layers_to_train>len(model.layers)):
        raise ValueError("Cannot train on more layers than the model!")
    if(outputs):
        num_classes = outputs
        model.layers.pop()
        x = Dense(num_classes, activation='softmax', name='output_predictions')(model.layers[-1].output)
        model = Model(input=model.input, output=[x])


    for layer in model.layers[:-layers_to_train]:
        layer.trainable = False

    model.summary()

    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    model.fit(X_train, Y_train,
               batch_size=batch_size,
               nb_epoch=nb_epoch,
               shuffle=True,
               verbose=1,
               validation_data=(X_valid, Y_valid),
               )

    return model



if __name__ == '__main__':
    img_rows, img_cols = 224, 224
    batch_size = 16
    nb_epoch = 2

    model = VGG16(include_top=True, weights='imagenet')
    sgd = SGD(lr=1e-3, decay=1e-6, momentum=0.9, nesterov=True)
    X_train, Y_train, X_valid, Y_valid = load_cifar10_data(img_rows, img_cols)

    loss = 'categorical_crossentropy'
    metrics = ['accuracy']

    X_train_new = X_train[:100, :, :, :]
    Y_train_new = Y_train[:100, :]
    X_valid_new = X_valid[:30, :, :, :]
    Y_valid_new = Y_valid[:30, :]

    transferLearn(model,1,batch_size, nb_epoch,sgd,loss,metrics,X_train_new,Y_train_new,X_valid_new,Y_valid_new)

    # Make predictions
    predictions_valid = model.predict(X_valid, batch_size=batch_size, verbose=1)

    # Cross-entropy loss score
    score = log_loss(Y_valid, predictions_valid)
    print(score)
