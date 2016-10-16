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

