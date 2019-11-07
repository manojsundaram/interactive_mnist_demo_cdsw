from PIL import Image
import numpy as np
from io import BytesIO
import base64
from PIL import ImageFilter
import os
import sys
#from keras.models import load_model
from pandas.io.json import dumps as jsonify
import torch
import torchvision

#import PIL.ImageOps   

model = torch.load('models/torch_model.pkl')

#example
#
#{
#  "image": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABwAAAAcCAYAAAByDd+UAAAARUlEQVRIS+3SwQkAQAwCQdN/0XclyELIa/MVEYZMkpfDGwe3tSXdFo2kkmIBnwaTtYKkTQjnkmKyVpC0CeFcUkzWCuekH906HAF1NLazAAAAAElFTkSuQmCC"
#}


def predict(args):
  im = Image.open(BytesIO(base64.b64decode(args['image'][22:])))
  im_inverted = im
  im_blur = im.filter(ImageFilter.GaussianBlur(5))
  im_small = im_blur.resize((28,28),Image.ANTIALIAS)
  im_bitmap = im_small.convert("L")
  #im_bitmap_inverted = PIL.ImageOps.invert(im_bitmap)
  im_array = np.asarray(im_bitmap)
  im_reshape = im_array.reshape(1,784)
  im_float = im_reshape.astype('float32')
  im_normalised =  -1+((im_float/255)*2)
  im_tensor = torch.tensor(im_normalised)
  with torch.no_grad():
    logps = model(im_tensor)
  ps = torch.exp(logps)
  probab = list(ps.cpu().numpy()[0])
  predict_val = probab.index(max(probab))
  return jsonify({'prediction': predict_val})