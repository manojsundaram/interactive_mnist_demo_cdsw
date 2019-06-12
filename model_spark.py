from flask import Flask,send_from_directory,request,send_file
import logging
from pandas.io.json import dumps as jsonify
from PIL import Image
from scipy.misc import imread
import numpy as np
from io import BytesIO
import base64
from PIL import ImageFilter
from numpy import zeros, newaxis

def predict(args):
  im = Image.open(BytesIO(base64.b64decode(args['image'][22:])))
  im_blur = im.filter(ImageFilter.GaussianBlur(5))
  im_small = im_blur.resize((28,28),Image.ANTIALIAS)
  im_bitmap = im_small.convert("L")
  im_array = np.asarray(im_bitmap)
  im_reshape = im_array.reshape(1,784).astype('int')
  #This was the tricky part
  im_df = spark.createDataFrame(pandas.DataFrame([tuple(np.insert(im_reshape,0,0))],columns=names))
  out = model.transform(im_df).select(['prediction']).collect()
  return jsonify({"prediction":int(out[0][0])})
