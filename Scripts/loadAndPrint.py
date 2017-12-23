import os
import sys
sys.path.insert(0, os.path.realpath('../'))
from keras.models import load_model


#MODEL_METADATA =['../../Experiment1Resnet.h5','../../Experiment2Resnet.h5','../../Experiment3Resnet.h5']
MODEL_METADATA =['../SavedData/Experiment1Resnet.h5','../SavedData/Experiment2Resnet.h5','../SavedData/Experiment3Resnet.h5']


if(len(sys.argv)>1):
    list_to_iterate = sys.argv[1:]

else:
    list_to_iterate = MODEL_METADATA

for model_file in list_to_iterate:
    print("Loading the model from {} ...".format(model_file))
    model = load_model(model_file)

    model.summary()