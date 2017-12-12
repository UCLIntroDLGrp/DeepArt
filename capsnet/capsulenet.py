"""
Keras implementation of CapsNet in Hinton's paper Dynamic Routing Between Capsules.
The current version maybe only works for TensorFlow backend. Actually it will be straightforward to re-write to TF code.
Adopting to other backends should be easy, but I have not tested this.
Usage:
       python CapsNet.py
       python CapsNet.py --epochs 100
       python CapsNet.py --epochs 100 --num_routing 3
       ... ...

Result:
    Validation accuracy > 99.5% after 20 epochs. Still under-fitting.
    About 110 seconds per epoch on a single GTX1070 GPU card

Author: Xifeng Guo, E-mail: `guoxifeng1990@163.com`, Github: `https://github.com/XifengGuo/CapsNet-Keras`
"""
import sys
import os
sys.path.insert(0, os.path.realpath('../'))
from keras import layers, models
from keras import backend as K
#from keras.utils import to_categorical
import numpy as np
from capsnet.capsulelayers import CapsuleLayer, PrimaryCap, Length, Mask
from keras.preprocessing.image import ImageDataGenerator


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
    conv1 = layers.Conv2D(filters=256, kernel_size=9, strides=1, padding='valid', activation='relu', name='conv1')(x)

    # Layer 2: Conv2D layer with `squash` activation, then reshape to [None, num_capsule, dim_vector]
    primarycaps = PrimaryCap(conv1, dim_vector=8, n_channels=32, kernel_size=9, strides=2, padding='valid')

    # Layer 3: Capsule layer. Routing algorithm works here.
    digitcaps = CapsuleLayer(num_capsule=n_class, dim_vector=16, num_routing=num_routing, name='digitcaps')(primarycaps)

    # Layer 4: This is an auxiliary layer to replace each capsule with its length. Just to match the true label's shape.
    # If using tensorflow, this will not be necessary. :)
    out_caps = Length(name='out_caps')(digitcaps)

    #last_layer_neuron_number = np.prod(input_shape)
    # Decoder network.
   # y = layers.Input(shape=(n_class,))
    #masked = Mask()([digitcaps, y])  # The true label is used to mask the output of capsule layer.
    #x_recon = layers.Dense(512, activation='relu')(masked)
    #x_recon = layers.Dense(1024, activation='relu')(x_recon)
    #x_recon = layers.Dense(last_layer_neuron_number, activation='sigmoid')(x_recon)
    #x_recon = layers.Reshape(target_shape=input_shape, name='out_recon')(x_recon)

    # two-input-two-output keras Model
    return models.Model(x, [out_caps])#, x_recon


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


    # callbacks
    #log = callbacks.CSVLogger(args.save_dir + '/log.csv')
    #tb = callbacks.TensorBoard(log_dir=args.save_dir + '/tensorboard-logs',
     #                          batch_size=args.batch_size, histogram_freq=args.debug)
    #checkpoint = callbacks.ModelCheckpoint(args.save_dir + '/weights-{epoch:02d}.h5',
      #                                     save_best_only=True, save_weights_only=True, verbose=1)
    #lr_decay = callbacks.LearningRateScheduler(schedule=lambda epoch: 0.001 * np.exp(-epoch / 10.))

    # compile the model
    print("compiling model")
    model.compile(optimizer='adam',
                  loss=[margin_loss], # 'mse'
                  metrics={'out_caps': 'accuracy'}) #loss_weights=[1., args.lam_recon],

    if(not args.augment_data):
        print("Start training without augmentation")
        # Training without data augmentation:
        history =  model.fit(x_train, y_train, batch_size=args.batch_size, epochs=args.epochs,
                  validation_data=(x_test, y_test)) #callbacks=[log, tb, checkpoint]

    else:
        print("train with augmentation")
        # -----------------------------------Begin: Training with data augmentation -----------------------------------#
        def train_generator(x, y, batch_size, shift_fraction=0.):
            train_datagen = ImageDataGenerator(width_shift_range=shift_fraction,
                                               height_shift_range=shift_fraction)  # shift up to 2 pixel for MNIST
            generator = train_datagen.flow(x, y, batch_size=batch_size)
            while 1:
                x_batch, y_batch = generator.next()
                yield ([x_batch, y_batch], [y_batch, x_batch])

        # Training with data augmentation. If shift_fraction=0., also no augmentation.
        model.fit_generator(generator=train_generator(x_train, y_train, args.batch_size, args.shift_fraction),
                            steps_per_epoch=int(y_train.shape[0] / args.batch_size),
                            epochs=args.epochs,
                            validation_data=[[x_test, y_test], [y_test, x_test]],
                            ) #callbacks=[log, tb, checkpoint, lr_decay]
        # -----------------------------------End: Training with data augmentation -----------------------------------#

    return model, history
