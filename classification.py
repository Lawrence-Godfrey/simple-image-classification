import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as pyplot

data = tf.keras.datasets.fashion_mnist
(train_images, train_labells),(test_images, test_labels) = data.load_data()

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

pyplot.imshow(train_images[7],cmap=pyplot.cm.binary)
pyplot.show()