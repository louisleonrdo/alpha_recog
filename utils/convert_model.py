import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import tensorflow as tf
from keras.models import load_model

# Load the .h5 model
h5_model = load_model('model/ConvModel.h5')

# Save the model in .keras format
h5_model.save('model/ConvModel.keras', save_format='keras')
