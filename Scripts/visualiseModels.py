
from keras.models import load_model
from vis.utils import utils
from vis.visualization import  visualize_activation
from matplotlib import pyplot as plt
from keras import activations
from vis.input_modifiers import Jitter



model = load_model('../SavedData/Experiment3Resnet.h5')
output_classes = 8
selected_indices = [361,213,206,151,251]



###### Visualising the output class activations
layer_index = utils.find_layer_idx(model,'output_predictions')
print(layer_index)
model.layers[layer_index].activation = activations.linear
model = utils.apply_modifications(model)

for filter_indices in range(0,output_classes):

    visualisation = visualize_activation(model,layer_index, filter_indices);
    img = visualize_activation(model, layer_index, filter_indices, max_iter=500, input_modifiers=[Jitter(16)])
    #img = Image.fromarray(visualisations, 'RGB')
    #img.show()
    plt.imshow(visualisation)
###################



###### Visualising 5 kernels of the activation 13 layer

layer_index = utils.find_layer_idx(model, 'activation_13')
vis_images = []
for filter_indices in selected_indices:
    img = visualize_activation(model, layer_index, filter_indices)
    # Utility to overlay text on image.
    visualisation = utils.draw_text(img, 'Filter {}'.format(filter_indices))
    vis_images.append(visualisation)

#Generate stitched image palette with 5 cols so we get 2 rows.
stitched = utils.stitch_images(vis_images, cols=5)
plt.figure()
plt.axis('off')
plt.imshow(stitched)
plt.show()
#############
