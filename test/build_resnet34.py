from __future__ import print_function
import caffe
from caffe.model_libs import *
from google.protobuf import text_format

import math
import os
import shutil
import stat
import subprocess
import sys


scale_args = {'bias_term': True}

def ResBlock(net, from_layer, blockName, channel_in, channel_out, layerNumber, reduceSize=True):
    stride = 1;
    if (reduceSize):
        stride = 2;

    for i in xrange(1, layerNumber + 1):
        ## basic body
        convName = '{}_L{}_conv1'.format(blockName,i)
        net[convName] = L.Convolution(net[from_layer],
                                num_output=channel_out,
                                kernel_size = 3,
                                stride = stride,
                                pad = 1)
        bnName = '{}_L{}_bn1'.format(blockName, i)
        net[bnName] = L.BatchNorm(net[convName], in_place=True)
        scaleName = '{}_L{}_scale1'.format(blockName, i)
        net[scaleName] = L.Scale(net[bnName], in_place=True, **scale_args)
        reluName = '{}_L{}_relu1'.format(blockName, i)
        net[reluName] = L.ReLU(net[scaleName], in_place=True)
        convName = '{}_L{}_conv2'.format(blockName,i)
        net[convName] = L.Convolution(net[reluName],
                                num_output=channel_out,
                                kernel_size = 3,
                                stride = 1,
                                pad = 1)
        bnName = '{}_L{}_bn2'.format(blockName, i)
        net[bnName] = L.BatchNorm(net[convName], in_place=True)
        scaleName = '{}_L{}_scale2'.format(blockName, i)
        net[scaleName] = L.Scale(net[bnName], in_place=True, **scale_args)
        shortcut1 = net[scaleName]

        ## shortcut
        shortcut2 = net[from_layer]
        if ( channel_in != channel_out ):
            convName = '{}_L{}_branch_conv1'.format(blockName, i)
            net[convName] = L.Convolution(net[from_layer],
                                num_output=channel_out,
                                kernel_size = 1,
                                stride = stride,
                                pad = 0)
            shortcut2 = net[convName]

        ## combin
        addName = '{}_L{}_res_add'.format(blockName, i)
        net[addName] = L.Eltwise(shortcut1, shortcut2)
        reluName = '{}_L{}_res_relu'.format(blockName, i)
        net[reluName] = L.ReLU(net[addName], in_place=True)

        from_layer = reluName
        stride = 1
        channel_in = channel_out

    return from_layer

def ResNet34Body(net, from_layer):
    net.conv1 = L.Convolution(net[from_layer],
                              num_output=64,
                              kernel_size=7,
                              stride=2,
                              pad=3)
    ## first level
    net.bn1 = L.BatchNorm(net.conv1, in_place=True)
    net.scale1 = L.Scale(net.bn1, in_place=True, **scale_args)
    net.relu1 = L.ReLU(net.scale1, in_place=True)
    net.pool1 = L.Pooling(net.conv1, pool=P.Pooling.MAX, kernel_size=3, stride=2, pad=1)

    ## 4 blocks
    last_layer = ResBlock(net, 'pool1', 'resa', 64, 64, 2, False);
    last_layer = ResBlock(net, last_layer, 'resb', 64, 128, 4, True);
    last_layer = ResBlock(net, last_layer, 'resc', 128, 256, 6, True);
    last_layer = ResBlock(net, last_layer, 'resd', 256, 512, 3, True);

    return net

model_name = "resnet34"
input_param = {'shape' : {
                'dim' :[1,3,224,224]
              }};


# Create train net.
net = caffe.NetSpec()
net.data = L.Input(input_param = input_param)

ResNet34Body(net, from_layer='data')

train_net_file = 'resnet-34.prototxt'
with open(train_net_file, 'w') as f:
    print('name: "{}"'.format(model_name), file=f)
    print(net.to_proto(), file=f)
sys.exit()



