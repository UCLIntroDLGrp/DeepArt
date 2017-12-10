import os
import sys
sys.path.insert(0, os.path.realpath('../'))
from keras.models import load_model

from Visualisation.visualisation import visualiseDenseLayer, visualiseCovolutionLayer

print("Loading the model in Experiment3Resnet.h5....")
model = load_model('../SavedData/Experiment3Resnet.h5')

print("Visualising the dense layer....")
selected_layer ='output_predictions'
output_classes = 8
save_name = "Experiment_3_Resnet_Output_Visualisation.png"
visualiseDenseLayer(model,selected_layer,output_classes, False, save=save_name)


print("Visualising the convolutional layer")
selected_layer = 'activation_13'
selected_indices = [361,213,206,151,251]
save_name = "Experiment_3_Resnet_Activation_13_Layer_Visualisation.png"
visualiseCovolutionLayer(model,selected_layer,selected_indices, False, save= save_name)


