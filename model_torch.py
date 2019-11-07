from PIL import Image, ImageFilter
import numpy as np
from io import BytesIO
import base64
from pandas.io.json import dumps as jsonify
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4*4*50, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4*4*50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


model = Net()
model.load_state_dict(torch.load('models/torch_model.pkl'))
model.eval()

#example
#
args = {
  "image": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABwAAAAcCAYAAAByDd+UAAAARUlEQVRIS+3SwQkAQAwCQdN/0XclyELIa/MVEYZMkpfDGwe3tSXdFo2kkmIBnwaTtYKkTQjnkmKyVpC0CeFcUkzWCuekH906HAF1NLazAAAAAElFTkSuQmCC"
}


def predict(args):
  im = Image.open(BytesIO(base64.b64decode(args['image'][22:])))
  im_blur = im.filter(ImageFilter.GaussianBlur(5))
  im_small = im_blur.resize((28,28),Image.ANTIALIAS)
  im_bitmap = im_small.convert("L")
  im_array = np.asarray(im_bitmap)
  im_float = im_array.astype('float32')
  im_normalised = (im_float/256)
  im_tensor = torch.tensor(im_normalised)
  im_tensor_shaped = im_tensor.reshape(1,1,28,28)
  with torch.no_grad():
    logps = model(im_tensor_shaped)
  ps = torch.exp(logps)
  probab = list(ps.cpu().numpy()[0])
  predict_val = probab.index(max(probab))
  return jsonify({'prediction': predict_val})