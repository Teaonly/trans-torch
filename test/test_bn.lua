require('nn')
require('transtorch')

torch.setdefaulttensortype('torch.FloatTensor')

local net = nn.Sequential()
net:add(nn.SpatialConvolutionMM(1, 20, 5, 5) )
net:add(nn.SpatialMaxPooling(2,2,2,2))
net:add(nn.SpatialConvolutionMM(20, 50, 5, 5))
net:add(nn.SpatialMaxPooling(2,2,2,2))

local bn = nn.SpatialBatchNormalization(50, 0.00001)
bn.running_mean = torch.rand(50);
bn.running_var = torch.rand(50);
bn.bias = torch.ones(50);

net:add(bn)
net:add(nn.Reshape(50*4*4))
net:add(nn.Linear(50*4*4, 500))
net:add(nn.ReLU())
net:add(nn.Linear(500, 10))
net:add(nn.SoftMax())

local x = torch.ones(1, 28, 28) * 0.5
x[{{},{15,28},{}}] = 1
net:evaluate();
y = net:forward(x)
print(y)

local caffeNet = transTorch.loadCaffe('./bn.prototxt');
local l = net:get(1)
transTorch.toCaffe(l, caffeNet, "conv1")
l = net:get(3)
transTorch.toCaffe(l, caffeNet, "conv2")
l = net:get(5)
transTorch.toCaffe(l, caffeNet, {"bn1", "sc1"})
l = net:get(7)
transTorch.toCaffe(l, caffeNet, "ip1")
l = net:get(9)
transTorch.toCaffe(l, caffeNet, "ip2")

transTorch.writeCaffe(caffeNet, "bn.caffemodel")

