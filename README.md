# trans-torch

Translating Torch model to other framework such as Caffe, MxNet ... 

Torch is excelent learning&research platform for deep learning, but its model can't be easy deployed without Luajit host. 
This tool is a simple solution to this problem, translating Torch model to other easy deployed platform such as Caffe or Mxnet. 

Directly translating Torch to Caffe (or others) is hard , we must Torch code to another DLS(like Caffe's prototxt) or python code(like Mxnet, Tensorflow)
But translating parameters ( weights and bias) only between diffrent frmamework is an easy task. For most model in zoo, we need only translating follow modules or layers:

* Linear module's weight and bias
* Convolution module's weight and bias
* BatchNormlaization's mean and var , also including alpha and beta .


## 1. Translating to Caffe 

### building 

```
CAFFE_DIR=/home/path_to/caffe luarocks make
```

### how to use this module

```
require('nn')
require('transtorch')

torch.setdefaulttensortype('torch.FloatTensor')

local net = nn.Sequential()
net:add(nn.Linear(2, 6))
net:add(nn.ReLU())
net:add(nn.Linear(6, 3))
net:add(nn.SoftMax())

local x = torch.ones(2) * 0.5
y = net:forward(x)

print(y)

local caffeNet = transTorch.loadCaffe('./mlp.prototxt');

local l = net:get(1)
transTorch.toCaffe(l, caffeNet, "ip1")
l = net:get(3)
transTorch.toCaffe(l, caffeNet, "ip2")

transTorch.writeCaffe(caffeNet, "mlp.caffemodel")

```

The Caffe's module will work same as the model build by Torch. We can verify model by simple_test.cpp in test floder.
See more demo code in test floder.

## 2. Translating to MxNet (TODO)

