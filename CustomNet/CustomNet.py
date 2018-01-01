import sys
import os
sys.path.insert(0, os.path.realpath('../'))
from keras import layers, models

from keras.applications.resnet50 import ResNet50
from TransferLearning.transferLearning import refactorOutputs, setTrainableLayers, freezeLayersUpTo, fineTune
from Capsnet.capsulelayers import PrimaryCap, CapsuleLayer,Length


def customModel():
    inputShape = (224,224,3)
    resnet = ResNet50(include_top=False, weights='imagenet',input_shape=inputShape)

    resnet.summary()

    # Layer 2: Conv2D layer with `squash` activation, then reshape to [None, num_capsule, dim_vector]
    primarycaps = PrimaryCap(resnet.layers[-1].output, dim_vector=8, n_channels=32, kernel_size=9, strides=2, padding='valid')

    # Layer 3: Capsule layer. Routing algorithm works here.
    digitcaps = CapsuleLayer(num_capsule=7, dim_vector=16, num_routing=3, name='digitcaps')(
        primarycaps)

    # Layer 4: This is an auxiliary layer to replace each capsule with its length. Just to match the true label's shape.
    # If using tensorflow, this will not be necessary. :)
    out_caps = Length(name='out_caps')(digitcaps)

    # two-input-two-output keras Model
    return models.Model(resnet.input, [out_caps])  # , x_recon


custom = customModel()
custom.summary()
'''
    def CapsNet(input_shape, n_class, num_routing):
        """
        A Capsule Network on MNIST.
        :param input_shape: data shape, 4d, [None, width, height, channels]
        :param n_class: number of classes
        :param num_routing: number of routing iterations
        :return: A Keras Model with 2 inputs and 2 outputs
        """
        x = layers.Input(shape=input_shape)

        # Layer 1: Just a conventional Conv2D layer
        conv1 = layers.Conv2D(filters=256, kernel_size=9, strides=1, padding='valid', activation='relu', name='conv1')(
            x)

        # Layer 2: Conv2D layer with `squash` activation, then reshape to [None, num_capsule, dim_vector]
        primarycaps = PrimaryCap(conv1, dim_vector=8, n_channels=32, kernel_size=9, strides=2, padding='valid')

        # Layer 3: Capsule layer. Routing algorithm works here.
        digitcaps = CapsuleLayer(num_capsule=n_class, dim_vector=16, num_routing=num_routing, name='digitcaps')(
            primarycaps)

        # Layer 4: This is an auxiliary layer to replace each capsule with its length. Just to match the true label's shape.
        # If using tensorflow, this will not be necessary. :)
        out_caps = Length(name='out_caps')(digitcaps)

        # two-input-two-output keras Model
        return models.Model(x, [out_caps])  # , x_recon
'''