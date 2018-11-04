from keras.applications.xception import Xception
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
from keras.models import Sequential, Model
from keras.layers import Dense, GlobalAveragePooling2D
import numpy as np

np.random.seed(7)

base_model = Xception(include_top=True, weights='imagenet', input_tensor=None, input_shape=(299,299,3))
xception_model = Model(inputs=base_model.input, outputs = base_model.get_layer('avg_pool').output)


#--------- Open dataset ----------

base_path = '/home/suayder/Desktop/transferLearningWisard/RandomDataSet/positivetest/'
with open('listNames/positivetest.txt') as fp:
    image_name = fp.readline()
    while image_name:
        image_path = base_path+image_name.strip()

        img = image.load_img(image_path, target_size=(299,299,3), interpolation='bilinear')
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis = 0)
        x = preprocess_input(x)

        features = xception_model.predict(x)
        md = np.mean(features)
        features = (features>=md)
        features = features.astype(int)
        features = np.reshape(features, (features.size,))

        f = open('featuresFromxCeption/positiveTest.txt', 'a')
        f.write(" ".join(str(el) for el in features)+"\n")
        f.close()
        image_name = fp.readline()