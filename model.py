from PIL import Image
import numpy as np
from io import BytesIO
import base64
from PIL import ImageFilter
import tensorflow as tf
import os
import sys
from keras.models import load_model
from pandas.io.json import dumps as jsonify

model = load_model('cnn_no_gpu.h5')

def predict(args):
  im = Image.open(BytesIO(base64.b64decode(args['image'][22:])))
  im_blur = im.filter(ImageFilter.GaussianBlur(5))
  im_small = im_blur.resize((28,28),Image.ANTIALIAS)
  im_bitmap = im_small.convert("L")
  im_array = np.asarray(im_bitmap)
  im_reshape = im_array.reshape(1,28,28,1)
  im_float = im_reshape.astype('float32')
  im_normalised =  im_float/255
  out = model.predict(im_normalised)
  predict_val = np.argmax(out)
  return jsonify({'prediction': predict_val})