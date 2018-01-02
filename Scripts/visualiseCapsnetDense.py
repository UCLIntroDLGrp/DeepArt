import os
import sys
sys.path.insert(0, os.path.realpath('../'))
from Visualisation.layerVisualisation import visualiseDenseLayer
from Capsnet.capsulenet import CapsNet
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


MODEL_METADATA =[('../SavedData/Experiment1CapsnetWeights.h5',"Experiment_1_Capsnet4040_Output_Visualisation.png")]

num_classes = 7
num_routing = 3
input_shape = [40,40,3]

for model_file, figure_save_name in MODEL_METADATA:
    print("Loading the model from {}...".format(model_file))
    model = CapsNet(input_shape=input_shape,
                n_class=num_classes,
                num_routing=num_routing)

    model.load_weights(model_file)

    print("Visualising the dense layer...")
    selected_layer = 'out_caps'
    output_classes = 7
    visualiseDenseLayer(model,selected_layer,output_classes, False, save=figure_save_name)
