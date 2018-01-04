import sys
import os
import numpy as np
from keras.models import load_model
from Preprocessing.preprocessing import crop_data_from_load2

sys.path.insert(0, os.path.realpath('../'))

os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"
debug=False
SAVE_DIRECTORY = "../SavedData/"

#Load the test data
if(not debug):
    X_test = np.load("../../../../../ml/2017/DeepArt/SavedData/X_test.npy")
    Y_test = np.load("../../../../../ml/2017/DeepArt/SavedData/Y_test.npy")
    X_test, Y_test = crop_data_from_load2(X_test, Y_test, (224, 224), 2)
else:
    #Debugging data
    X_test = np.load("../SavedData/X_train.npy")
    Y_test = np.load("../SavedData/Y_train.npy")
    X_test = X_test[:128]
    Y_test = Y_test[:128]

#Models to test
if(not debug):
    MODEL_METADATA =[ ('../SavedData/Experiment2Resnet2.h5',"Experiment2_Resnet_Test_Result.npy"),
                    ('../../Experiment1Resnet.h5',"Experiment1_Resnet_Test_Result.npy"),
                      ('../../Experiment2Resnet.h5',"Experiment2_Resnet_Test_Result.npy"),
                     ('../../Experiment3Resnet.h5',"Experiment3_Resenet_Test_Result.npy")]
else:
    #Debugging data
     MODEL_METADATA =[('../SavedData/Experiment1Resnet.h5',"Experiment1_Resnet_Test_Result.npy"),
                      ('../SavedData/Experiment2Resnet.h5',"Experiment2_Resnet_Test_Result.npy"),
                     ('../SavedData/Experiment3Resnet.h5',"Experiment3_Resenet_Test_Result.npy")]


#MODEL_METADATA = [("../SavedData/Experiment1Capsnet.h5","Experiment1_Capsnet_Test_Result.npy")]


#For each of the models
for model_file, test_value_save_name in MODEL_METADATA:
    #Load the model
    print("Loading the model from {} ...".format(model_file))
    model = load_model(model_file)

    print("Evaluating on test set...")
    #evaluate performance on the test set
    loss_and_acc = model.evaluate(X_test,Y_test)

    #Print and save
    print("Loss and accuracy on the test sets for model {} :".format(test_value_save_name))
    print(loss_and_acc)
    np_loss_and_acc = np.asarray(loss_and_acc)
    np.save(SAVE_DIRECTORY+test_value_save_name,np_loss_and_acc)