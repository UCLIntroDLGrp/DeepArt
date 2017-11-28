#Credit for the code and data to Francois Chollet, https://github.com/fchollet
# Code from : https://github.com/fchollet/deep-learning-models + releases
#
#
###

'''VGG16 model for Keras.
# Reference:

- [Very Deep Convolutional Networks for Large-Scale Image Recognition](https://arxiv.org/abs/1409.1556)

'''
from __future__ import print_function


from keras.models import Model
from keras.layers import Flatten, Dense, Input
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K

TF_WEIGHTS_PATH = '../pretrainedWeights/vgg16_weights_tf_dim_ordering_tf_kernels.h5'
TF_WEIGHTS_PATH_NO_TOP = '../pretrainedWeights/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5'


def VGG16(include_top=True, weights=None,
          input_tensor=None):
    '''Instantiate the VGG16 architecture,
    optionally loading weights pre-trained
    on ImageNet.
    # Arguments
        include_top: whether to include the 3 fully-connected
            layers at the top of the network.
        weights: one of `None` (random initialization)
            or "imagenet" (pre-training on ImageNet).
        input_tensor: optional Keras tensor (i.e. output of `layers.Input()`)
            to use as image input for the model.

    # Returns
        A Keras model instance.
    '''
    if weights not in {'imagenet', None}:
        raise ValueError('The `weights` argument should be either '
                         '`None` (random initialization) or `imagenet` '
                         '(pre-training on ImageNet).')
    # Determine proper input shape
    if include_top:
        input_shape = (224, 224, 3)
    else:
        input_shape = (None, None, 3)

    if input_tensor is None:
        img_input = Input(shape=input_shape)
    else:
        if not K.is_keras_tensor(input_tensor):
            img_input = Input(tensor=input_tensor)
        else:
            img_input = input_tensor
    # Block 1
    x = Conv2D(64, (3, 3), activation="relu", name="block1_conv1", padding="same")(img_input)
    x = Conv2D(64, (3, 3), activation="relu", name="block1_conv2", padding="same")(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

    # Block 2
    x = Conv2D(128, (3, 3), activation="relu", name="block2_conv1", padding="same")(x)
    x = Conv2D(128, (3, 3), activation="relu", name="block2_conv2", padding="same")(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

    # Block 3
    x = Conv2D(256, (3, 3), activation="relu", name="block3_conv1", padding="same")(x)
    x = Conv2D(256, (3, 3), activation="relu", name="block3_conv2", padding="same")(x)
    x = Conv2D(256, (3, 3), activation="relu", name="block3_conv3", padding="same")(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

    # Block 4
    x = Conv2D(512, (3, 3), activation="relu", name="block4_conv1", padding="same")(x)
    x = Conv2D(512, (3, 3), activation="relu", name="block4_conv2", padding="same")(x)
    x = Conv2D(512, (3, 3), activation="relu", name="block4_conv3", padding="same")(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

    # Block 5
    x = Conv2D(512, (3, 3), activation="relu", name="block5_conv1", padding="same")(x)
    x = Conv2D(512, (3, 3), activation="relu", name="block5_conv2", padding="same")(x)
    x = Conv2D(512, (3, 3), activation="relu", name="block5_conv3", padding="same")(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)

    if include_top:
        # Classification block
        x = Flatten(name='flatten')(x)
        x = Dense(4096, activation='relu', name='fc1')(x)
        x = Dense(4096, activation='relu', name='fc2')(x)
        x = Dense(1000, activation='softmax', name='predictions')(x)

    # Create model
    model = Model(img_input, x)

    # load weights
    if weights == 'imagenet':
        if include_top:
            weights_path = TF_WEIGHTS_PATH
        else:
            weights_path = TF_WEIGHTS_PATH_NO_TOP
        model.load_weights(weights_path)

    return model


if __name__ == '__main__':
    model = VGG16(include_top=True, weights='imagenet')
