#Load CNN and predict
from PIL import Image, ImageFilter, ImageOps
#from scipy.misc import imread, imresize
import numpy as np
import torch


im = Image.open('test3.png')# Image.open(BytesIO(base64.b64decode(args['image'][22:])))
#im_inverted = im
#im_blur = im.filter(ImageFilter.GaussianBlur(5))
#im_small = im_blur.resize((28,28),Image.ANTIALIAS)
im_bitmap = im.convert("L")
im_bitmap_inverted = ImageOps.invert(im_bitmap)
im_array = np.asarray(im_bitmap_inverted)
im_reshape = im_array.reshape(1,784)
im_float = im_reshape.astype('float32')
im_normalised =  -1+((im_float/255)*2)
im_tensor = torch.tensor(im_normalised)

model = torch.load('models/torch_model.pkl')


with torch.no_grad():
  logps = model(im_tensor)
ps = torch.exp(logps)
probab = list(ps.cpu().numpy()[0])
predict_val = probab.index(max(probab))
print(predict_val)


#jsonify({'prediction': predict_val})



##
#
##x = imread()
##compute a bit-wise inversion so black becomes white and vice versa
#x = np.invert(x)
##make it the right size
#x = imresize(x,(28,28))
##convert to a 4D tensor to feed into our model
#x = x.reshape(1,28,28,1)
#x = x.astype('float32')
#x /= 255
#
##perform the prediction
#import tensorflow as tf
#config = tf.ConfigProto(
#        device_count = {'GPU': 0}
#    )
#sess = tf.Session(config=config)
#from keras.models import load_model
#
#
#model = load_model('cnn_new.h5')
#out = model.predict(x)
#print(np.argmax(out))
#
