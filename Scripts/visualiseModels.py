import os
import sys
sys.path.insert(0, os.path.realpath('../'))
from keras.models import load_model

from Visualisation.visualisation import visualiseDenseLayer, visualiseCovolutionLayer


model = load_model('../SavedData/Experiment3Resnet.h5')

selected_layer ='output_predictions'
output_classes = 8
visualiseDenseLayer(model,selected_layer,output_classes)


selected_layer = 'activation_13'
selected_indices = [361,213,206,151,251]
visualiseCovolutionLayer(model,selected_layer,selected_indices)


