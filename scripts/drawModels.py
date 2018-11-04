from keras.applications.xception import Xception
from keras.models import Sequential, Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras.utils import plot_model
import numpy as np

np.random.seed(7)

base_model = Xception(include_top=True, weights='imagenet', input_tensor=None, input_shape=(299,299,3))
xception_model = Model(inputs=base_model.input, outputs = base_model.get_layer('avg_pool').output)
plot_model(xception_model, to_file='model.png')