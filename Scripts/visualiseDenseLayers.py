import os
import sys
sys.path.insert(0, os.path.realpath('../'))
from keras.models import load_model
from Visualisation.layerVisualisation import visualiseDenseLayer, visualiseCovolutionLayer


MODEL_METADATA =[('../SavedData/Experiment3Resnet.h5',"Experiment_3_Resnet_Output_Visualisation.png")]

for model_file, figure_save_name in MODEL_METADATA:
    print("Loading the model from {}...".formal(model_file))
    model = load_model(model_file)

    print("Visualising the dense layer...")
    selected_layer = 'output_predictions'
    output_classes = 8
    visualiseDenseLayer(model,selected_layer,output_classes, False, save=figure_save_name)
