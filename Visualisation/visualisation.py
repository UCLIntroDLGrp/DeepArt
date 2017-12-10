from vis.utils import utils
from vis.visualization import  visualize_activation
from matplotlib import pyplot as plt
from keras import activations
from vis.input_modifiers import Jitter


def visualiseDenseLayer(model,layer_name,output_classes, verbose, save=False):
    '''
    Makes a plot for visualising  the activations of a dense layer.
    :param model:  The model to visualise.
    :param layer_name: The name of the dense layer to visualise.
    :param output_classes: The number of activations in the layer.
    :param verbose: Print statements of progress.
    :return: N/A
    '''
    layer_index = utils.find_layer_idx(model, layer_name)
    model.layers[layer_index].activation = activations.linear
    model = utils.apply_modifications(model)

    vis_images = []
    for filter_index in range(0, output_classes):
        if(verbose):
            print("Preparing Visualisation for class {} in layer {}".format(filter_index,layer_name))
        visualisation = visualize_activation(model, layer_index, filter_index, max_iter=500, input_modifiers=[Jitter(16)])

        img = utils.draw_text(visualisation, 'Class {}'.format(filter_index))
        vis_images.append(img)


    stitched = utils.stitch_images(vis_images, cols=4)
    plt.figure()
    plt.axis('off')
    plt.imshow(stitched)
    plt.show()
    if(save):
        plt.savefig("../SavedData/"+ save)


def visualiseCovolutionLayer(model,selected_layer,selected_indices, verbose,save=False):
    '''
    Makes a plot for visualising  the activations of a convolutional layer.
    :param model: The model to visualise.
    :param selected_layer: The name of the dense layer to visualise.
    :param selected_indices: The indices of the filters to visualise.
    :param verbose: Print statements of progress.
    :return: N/A.
    '''
    layer_index = utils.find_layer_idx(model, selected_layer)
    vis_images = []
    for filter_index in selected_indices:
        if (verbose):
            print("Preparing Visualisation for filter {} in layer {}".format(filter_index,selected_layer))
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
    if(save):
        plt.savefig("../SavedData/" + save)

