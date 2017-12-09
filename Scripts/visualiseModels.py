
from keras.models import load_model
from vis.utils import utils
from vis.visualization import visualize_saliency, visualize_activation
from PIL import Image

model = load_model('Experiment3Model.h5')


layer_index = utils.find_layer_idx(model,'block5_conv3')
print(layer_index)
ret = visualize_activation(model,layer_index, None, X_test[1,:,:,:]);

img = Image.fromarray(ret, 'RGB')
img.show()

ret2 = visualize_saliency(model, layer_index, None, X_test[1, :, :, :]);

img2 = Image.fromarray(ret2, 'RGB')
img2.show()
