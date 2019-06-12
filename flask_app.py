from flask import Flask,send_from_directory,request,send_file
import logging
from pandas.io.json import dumps as jsonify
from PIL import Image
from scipy.misc import imread
import numpy as np
from io import BytesIO
import base64
from PIL import ImageFilter
import tensorflow as tf
from numpy import zeros, newaxis
import tensorflow as tf
from IPython.display import Javascript,HTML
#import imageio
#from keras.models import load_model
# model = load_model('TFKeras.h5')
#model = tf.keras.models.load_model('my_tf_model.h5')
#model = load_model('my_tf_model.h5')

from keras.models import load_model
model = load_model('cnn_new.h5')
graph = tf.get_default_graph()

log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)

app= Flask(__name__,static_url_path='')
    
@app.route('/')
def index():
    return "<script> window.location.href = '/flask/mnist.html'</script>"

@app.route('/flask/<path:path>')
def send_file(path):
    return send_from_directory('flask', path)

@app.route('/test', methods=['POST'])
def post_test():
    global graph
    with graph.as_default():
        if request.json:
            im = Image.open(BytesIO(base64.b64decode(request.json['image'][22:])))
            im_blur = im.filter(ImageFilter.GaussianBlur(5))
            im_small = im_blur.resize((28,28),Image.ANTIALIAS)
            im_bitmap = im_small.convert("L")
            im_array = np.asarray(im_bitmap)
            im_reshape = im_array.reshape(1,28,28,1)
            im_float = im_reshape.astype('float32')
            im_normalised =  im_float/255
            out = model.predict(im_normalised)
            predict_val = np.argmax(out)
            return jsonify({"prediction":predict_val})
        else:
            return "failed"

HTML("<a href='https://{}.{}'>Open Web View</a>".
     format(os.environ['CDSW_ENGINE_ID'],os.environ['CDSW_DOMAIN']))

if __name__=="__main__":
  app.run(host=os.environ['CDSW_IP_ADDRESS'], 
          port=int(os.environ['CDSW_PUBLIC_PORT']))