import sys
import os
sys.path.insert(0, os.path.realpath('../'))
from Capsnet.capsulenet import CapsNet, train
import numpy as np
from Utilities.utilities import selectData
from Preprocessing.preprocessing import generate_cropped_training_and_test_data,crop_data_from_load
from pickle import dump
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2"

sm_train_data = False
debug_data = False

crop_dims = (100, 100)
number_of_crops = 4


if(sm_train_data):

    # Get the data:
    #directory = "../wikiart"
    directory = '../Art_Data_sm'
    validation_size = 10.0 / 100
    train_size = 80.0 / 100
    X_train, X_validation, Y_train, Y_validation = generate_cropped_training_and_test_data(directory,
                                                                               crop_dims,
                                                                               number_of_crops,
                                                                               validation_size,
                                                                               train_size,
                                                                               10)

else:
    if(not debug_data):
        X_train = np.load("../../../../../ml/2017/DeepArt/SavedData/X_train.npy")
        X_validation = np.load("../../../../../ml/2017/DeepArt/SavedData/X_validation.npy")
        Y_train = np.load("../../../../../ml/2017/DeepArt/SavedData/Y_train.npy")
        Y_validation = np.load("../../../../../ml/2017/DeepArt/SavedData/Y_validation.npy")

        X_train, X_validation, Y_train, Y_validation = crop_data_from_load(
            X_train, X_validation, Y_train, Y_validation, crop_dims, number_of_crops)

    else:
        X_train = np.load("../SavedData/X_train.npy")
        X_validation = np.load("../SavedData/X_validation.npy")
        Y_train = np.load("../SavedData/Y_train.npy")
        Y_validation = np.load("../SavedData/Y_validation.npy")
        X_train, X_validation, Y_train, Y_validation = crop_data_from_load(
            X_train, X_validation, Y_train, Y_validation, crop_dims, number_of_crops)


if (debug_data):
    # Select only few training examples - uncomment for quick testing
    X_train = selectData(X_train, 16)
    Y_train = selectData(Y_train, 16)
    X_validation = selectData(X_validation, 8)
    Y_validation = selectData(Y_validation, 8)

if(not debug_data):
    num_routing = 3
    batch_size = 32
    nb_epoch = 15
    num_classes = 7
    shift_fraction = 0
    debug = 0
    augment_data = False
else:
    num_routing = 3
    batch_size = 4
    nb_epoch = 1
    num_classes = 8
    shift_fraction = 0
    debug = 0
    augment_data = False

class Args(object):
    def __init__(self , num_routing,batch_size, nb_epoch,num_classes ,shift_fraction,debug, augment_data):#lam_recon,
        self.num_routing = num_routing
        self.batch_size = batch_size
        self.epochs = nb_epoch
        self.num_classes = num_classes
       # self.lam_recon = lam_recon
        self.shift_fraction = shift_fraction
        self.debug = debug
        self.augment_data = augment_data


args = Args(num_classes,batch_size,nb_epoch,num_classes,shift_fraction,debug,augment_data)#lam_recon


model = CapsNet(input_shape=[crop_dims[0], crop_dims[1], 3],
                n_class=num_classes,
                num_routing=num_routing)

model.summary()

model , history = train(model=model, data=((X_train, Y_train), (X_validation, Y_validation)), args=args)

print("Finishing Training")

model.save("../SavedData/Experiment1Capsnet100100.h5")
model.save_weights("../SavedData/Experiment1CapsnetWeights100100.h5")
pickle_out = open("../SavedData/Experiment1CapsnetHistory100100.pickle", "wb")
dump(history.history, pickle_out)
pickle_out.close()


'''
def test(model, data):
    x_test, y_test = data
    y_pred, x_recon = model.predict([x_test, y_test], batch_size=100)
    print('-'*50)
    print('Test acc:', np.sum(np.argmax(y_pred, 1) == np.argmax(y_test, 1))/y_test.shape[0])

    import matplotlib.pyplot as plt
    from utils import combine_images
    from PIL import Image

    img = combine_images(np.concatenate([x_test[:50],x_recon[:50]]))
    image = img * 255
    Image.fromarray(image.astype(np.uint8)).save("real_and_recon.png")
    print()
    print('Reconstructed images are saved to ./real_and_recon.png')
    print('-'*50)
    plt.imshow(plt.imread("real_and_recon.png", ))
    plt.show()





# setting the hyper parameters
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', default=100, type=int)
parser.add_argument('--epochs', default=20, type=int)
parser.add_argument('--lam_recon', default=0.0005, type=float)
parser.add_argument('--num_routing', default=3, type=int)  # num_routing should > 0
parser.add_argument('--shift_fraction', default=0.1, type=float)
parser.add_argument('--debug', default=0, type=int)  # debug>0 will save weights by TensorBoard
parser.add_argument('--save_dir', default='./result')
parser.add_argument('--is_training', default=1, type=int)
parser.add_argument('--weights', default=None)
args = parser.parse_args()
print(args)
if not os.path.exists(args.save_dir):
    os.makedirs(args.save_dir)

# load data
(x_train, y_train), (x_test, y_test) = load_mnist()

# define model
model = CapsNet(input_shape=[28, 28, 1],
                n_class=len(np.unique(np.argmax(y_train, 1))),
                num_routing=args.num_routing)
model.summary()
plot_model(model, to_file=args.save_dir+'/model.png', show_shapes=True)

# train or test
if args.weights is not None:  # init the model weights with provided one
    model.load_weights(args.weights)
if args.is_training:
    train(model=model, data=((x_train, y_train), (x_test, y_test)), args=args)
else:  # as long as weights are given, will run testing
    if args.weights is None:
        print('No weights are provided. Will test using random initialized weights.')
    test(model=model, data=(x_test, y_test))

'''
