import os
import sys
sys.path.insert(0, os.path.realpath('../'))
from Visualisation.layerVisualisation import visualiseGenericLayer
from CustomNet.CustomNet import customModel
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


MODEL_METADATA =[('../SavedData/Experiment1CustomNet.h5',"Experiment_1_customNetNoDropout_Output_Visualisation.png")]

num_classes = 7
input_shape = [224,224,3]

for model_file, figure_save_name in MODEL_METADATA:
    print("Loading the model from {}...".format(model_file))
    model = customModel(input_shape, num_classes,False)

    model.load_weights(model_file)

    print("Visualising the dense layer...")
    selected_layer = 'out_caps'
    output_classes = 7
    visualiseGenericLayer(model,selected_layer,output_classes, True, save=figure_save_name)
