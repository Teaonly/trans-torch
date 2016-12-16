local ffi = require('ffi')
local C = transTorch._C

local toLinear = function(tm, caffeNet, layerName) 
    --assert(tm.weight:type() == 'torch.FloatTensor')
    local weight = tm.weight:cdata()
    local bias = tm.bias:cdata()
    C.writeCaffeLinearLayer(caffeNet[0], layerName, weight, bias)
end

local toConv = function(tm, caffeNet, layerName)
    --assert(tm.weight:type() == 'torch.FloatTensor')
    local weights = tm.weight:float():cdata()
    local bias = tm.bias:float():cdata()
    C.writeCaffeConvLayer(caffeNet[0], layerName, weights, bias)
end

local toBatchNorm = function(tm, caffeNet, layerName)
    if ( tm.affine == true) then
        assert(type(layerName) == 'table')
        assert(#layerName == 2)
        local weights = tm.weight:float():cdata()
        local bias = tm.bias:float():cdata()
        local mean = tm.running_mean:float():cdata()
        local var = tm.running_var:float():cdata()
        C.writeCaffeBNLayer(caffeNet[0], layerName[1], mean, var);
        C.writeCaffeScaleLayer(caffeNet[0], layerName[2], weights, bias);
    else
        assert(type(layerName) == 'string')
        local mean = tm.running_mean:float():cdata()
        local var = tm.running_var:float():cdata()
        C.writeCaffeBNLayer(caffeNet[0], layerName[0], mean, var);
    end
end

transTorch.loadCaffe = function(prototxt_name, binary_name) 
    assert(type(prototxt_name) == 'string')
    if ( binary_name ~= nil ) then
        assert(type(binary_name) == 'string')
    end
    
    local net = ffi.new("void*[1]")  
    net[0] = C.loadCaffeNet(prototxt_name, binary_name) 
    
    return net
end

transTorch.releaseCaffe = function(net) 
    C.releaseCaffeNet(net[0]);
end

transTorch.writeCaffe = function(net, fileName)
    C.saveCaffeNet(net[0], fileName);
end

transTorch.toCaffe = function(tmodel, caffeNet, layerName)
    local mtype = torch.type(tmodel)
    if ( mtype == 'nn.Linear' ) then
        toLinear(tmodel, caffeNet, layerName)
    elseif ( mtype == 'nn.BatchNormalization' or mtype == 'nn.SpatialBatchNormalization' ) then
        toBatchNorm(tmodel, caffeNet, layerName)
    elseif ( string.match(mtype, 'Convolution') ) then
        toConv(tmodel, caffeNet, layerName)
    else
        print(" ##ERROR## unspported layer:" .. mtype)
        assert(false)
    end
end


