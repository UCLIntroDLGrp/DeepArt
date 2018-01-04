import sys
import os
sys.path.insert(0, os.path.realpath('../'))
import numpy as np
from Capsnet.capsulenet import margin_loss
from Capsnet.capsnetTesting import getAcc
from Capsnet.capsulenet import CapsNet
import tensorflow as tf
from Preprocessing.preprocessing import crop_data_from_load2



input_shape = [224,224,3]
os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3"
debug = False

def test(model, data):
    x_test, y_test = data
    y_pred = model.predict([x_test, y_test], batch_size=100)
    return y_pred


sys.path.insert(0, os.path.realpath('../'))


debug=True
SAVE_DIRECTORY = "../SavedData/"

#Load the test data
if(not debug):
    X_test = np.load("../../../../../ml/2017/DeepArt/SavedData/X_test.npy")
    Y_test = np.load("../../../../../ml/2017/DeepArt/SavedData/Y_test.npy")
else:
    #Debugging data
    X_test = np.load("../SavedData/X_train.npy")
    Y_test = np.load("../SavedData/Y_train.npy")
    X_test = X_test[:32]
    Y_test = Y_test[:32]

#Models to test
if(not debug):
    MODEL_METADATA = [("../SavedData/Experiment1Capsnet.h5","Experiment1_Capsnet_4040_Test_Result.npy",(40,40)),
                      ("../SavedData/Experiment1CapsnetWeights100100.h5 ", "Experiment1_Capsnet_100100_Test_Result.npy"),(100,100)]
else:
    #Debugging data
     MODEL_METADATA =  [("../SavedData/Experiment1CapsnetWeights.h5","Experiment1_Capsnet_Test_Result.npy")]


#For each of the models
for model_file, test_value_save_name, dims in MODEL_METADATA:
    #Load the model
    X_test, Y_test = crop_data_from_load2(X_test, Y_test, dims, 2)
    print("Loading the model from {} ...".format(model_file))

    if(not debug):
        num_classes =7
        num_routing = 3
    else:
        num_classes =8
        num_routing = 3

    #model = load_model(model_file)
    model = CapsNet(input_shape=input_shape,
                n_class=num_classes,
                num_routing=num_routing)


    model.load_weights(model_file)

    print("Evaluating on test set...")
    #get the predictions on the test set
    Y_pred = model.predict(X_test, batch_size=4)

    #get evaluation metrics
    with tf.Session() as sess:
        loss = margin_loss(Y_test,Y_pred).eval()
    acc = getAcc(Y_test,Y_pred)
    loss_and_acc=[loss, acc]

    #Print and save
    print("Loss and accuracy on the test sets for model {} :".format(test_value_save_name))
    print(loss_and_acc)
    np_loss_and_acc = np.asarray(loss_and_acc)
    np.save(SAVE_DIRECTORY+test_value_save_name,np_loss_and_acc)