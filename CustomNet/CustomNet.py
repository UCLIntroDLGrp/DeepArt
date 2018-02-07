import sys
import os
sys.path.insert(0, os.path.realpath('../'))
from keras import layers, models
from keras import backend as K
from keras.layers import Dropout

from keras.applications.resnet50 import ResNet50
from TransferLearning.transferLearning import refactorOutputs, setTrainableLayers, freezeLayersUpTo, fineTune
from Capsnet.capsulelayers import PrimaryCap, CapsuleLayer,Length


def customModel(inputShape, num_classes, dropout):
    resnet = ResNet50(include_top=False, weights='imagenet',input_shape=inputShape)
    resnet.layers.pop()

    freezeLayersUpTo(resnet,None)

    # Layer 2: Conv2D layer with `squash` activation, then reshape to [None, num_capsule, dim_vector]
    primarycaps = PrimaryCap(resnet.layers[-1].output, dim_vector=8, n_channels=32, kernel_size=4, strides=2, padding='valid')
    if(dropout):
        dropoutCaps = Dropout(0.5)(primarycaps)
        # Layer 3: Capsule layer. Routing algorithm works here.
        digitcaps = CapsuleLayer(num_capsule=num_classes, dim_vector=16, num_routing=3, name='digitcaps')(
            dropoutCaps)
    else:
        # Layer 3: Capsule layer. Routing algorithm works here.
        digitcaps = CapsuleLayer(num_capsule=num_classes, dim_vector=16, num_routing=3, name='digitcaps')(
            primarycaps)


    # Layer 4: This is an auxiliary layer to replace each capsule with its length. Just to match the true label's shape.
    # If using tensorflow, this will not be necessary. :)
    out_caps = Length(name='out_caps')(digitcaps)

    # two-input-two-output keras Model

    return models.Model(resnet.input, [out_caps])  # , x_recon





def margin_loss(y_true, y_pred):
    """
    Margin loss for Eq.(4). When y_true[i, :] contains not just one `1`, this loss should work too. Not test it.
    :param y_true: [None, n_classes]
    :param y_pred: [None, num_capsule]
    :return: a scalar loss value.
    """
    L = y_true * K.square(K.maximum(0., 0.9 - y_pred)) + \
        0.5 * (1 - y_true) * K.square(K.maximum(0., y_pred - 0.1))

    return K.mean(K.sum(L, 1))





def train(model, data, args):
    """
    Training a CapsuleNet
    :param model: the CapsuleNet model
    :param data: a tuple containing training and testing data, like `((x_train, y_train), (x_test, y_test))`
    :param args: arguments
    :return: The trained model
    """
    # unpacking the data
    (x_train, y_train), (x_test, y_test) = data

    # compile the model
    print("compiling model")
    model.compile(optimizer='adam',
                  loss=[margin_loss], # 'mse'
                  metrics={'out_caps': 'accuracy'}) #loss_weights=[1., args.lam_recon],

    print("Start training without augmentation")
    # Training without data augmentation:
    history =  model.fit(x_train, y_train, batch_size=args.batch_size, epochs=args.epochs,
              validation_data=(x_test, y_test)) #callbacks=[log, tb, checkpoint]



    return model, history

