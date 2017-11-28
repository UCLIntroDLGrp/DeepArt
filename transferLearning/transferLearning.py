from keras.layers import Dense
from keras.models import Model


def refactorOutputs(model,outputs,verbose=False):
    '''
    Replace the final layer of the input model with a fully connected layer with specified outputs.
    :param model:  The model to modify
    :param outputs:  The number of outputs in the new final fully connected layer.
    :param verbose: If True will print summary of the model before and after modification
    :return: The model, after the final layer has been replaced
    '''
    if(verbose):
        model.summary()

    num_classes = outputs
    model.layers.pop()
    x = Dense(num_classes, activation='softmax', name='output_predictions')(model.layers[-1].output)
    model = Model(inputs=model.input, outputs=x)

    if(verbose):
        model.summary()

    return model


def setTrainableLayers(model, layers_to_train ):
    '''
    Sets the number of layers that can be trained in the model.
    :param model: The model we want to train.
    :param layers_to_train: int, The number of layers to train from last to first.
    :return: The model with only the last layers_to_train that are trainable
    '''

    if(layers_to_train>len(model.layers)):
        raise ValueError("Cannot train on more layers than the model!")

    for layer in model.layers[:-layers_to_train]:
        layer.trainable = False

    return model

def fineTune(model,batch_size, nb_epoch,optimizer,loss,
                  metrics, X_train, Y_train, X_valid, Y_valid, verbose=False):
    '''
    Fine tunes the model.
    :param model: The model to fine-tune
    :param batch_size: int, The batch size to train on.
    :param nb_epoch: int, The number of epochs to train on.
    :param optimizer: The optimiser to use, from keras.optimizers.
    :param loss: The loss parameter in model.compile
    :param metrics: The metrics parameter in model.compile
    :param X_train: Training Data
    :param Y_train: Training Labels
    :param X_valid: Validation Data
    :param Y_valid: Validation Labels
    :param verbose: If true prints a model summary.
    :return: The model, after the trainable layers have been fine-tuned to the new data.
    '''

    if(verbose):
        model.summary()

    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    model.fit(X_train, Y_train,
               batch_size=batch_size,
               epochs=nb_epoch,
               shuffle=True,
               verbose=1,
               validation_data=(X_valid, Y_valid),
               )

    return model