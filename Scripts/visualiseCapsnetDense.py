import os
import sys
sys.path.insert(0, os.path.realpath('../'))
from keras.models import load_model
from Visualisation.layerVisualisation import visualiseDenseLayer, visualiseCovolutionLayer
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


MODEL_METADATA =[('../SavedData/Experiment2Resnet2.h5',"Experiment_2_Resnet2_Output_Visualisation.png")]

for model_file, figure_save_name in MODEL_METADATA:
    print("Loading the model from {}...".format(model_file))
    model = load_model(model_file)

    print("Visualising the dense layer...")
    selected_layer = 'output_predictions'
    output_classes = 7
    visualiseDenseLayer(model,selected_layer,output_classes, False, save=figure_save_name)
