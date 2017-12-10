from vis.utils import utils
from vis.visualization import  visualize_activation
from matplotlib import pyplot as plt
from keras import activations
from vis.input_modifiers import Jitter


def visualiseDenseLayer(model,layer_name,output_classes):
    layer_index = utils.find_layer_idx(model, layer_name)
    model.layers[layer_index].activation = activations.linear
    model = utils.apply_modifications(model)

    vis_images = []
    for filter_index in range(0, output_classes):
        visualisation = visualize_activation(model, layer_index, filter_index, max_iter=500, input_modifiers=[Jitter(16)])

        img = utils.draw_text(visualisation, 'Class {}'.format(filter_index))
        vis_images.append(img)


    stitched = utils.stitch_images(vis_images, cols=4)
    plt.figure()
    plt.axis('off')
    plt.imshow(stitched)
    plt.show()


def visualiseCovolutionLayer(model,selected_layer,selected_indices):
    layer_index = utils.find_layer_idx(model, selected_layer)
    vis_images = []
    for filter_index in selected_indices:
        img = visualize_activation(model, layer_index, filter_index)
        # Utility to overlay text on image.
        visualisation = utils.draw_text(img, 'Filter {} on layer {}'.format(filter_index,selected_layer))
        vis_images.append(visualisation)

    # Generate stitched image palette with 5 cols so we get 2 rows.
    stitched = utils.stitch_images(vis_images, cols=5)
    plt.figure()
    plt.axis('off')
    plt.imshow(stitched)
    plt.show()
    #############
