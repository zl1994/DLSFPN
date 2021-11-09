import torch
import torch.nn as nn
from collections import OrderedDict
from mmcv.cnn import kaiming_init, constant_init
from mmcv.runner import load_checkpoint

class _DenseLayer(nn.Sequential):
    def __init__(self, in_channels, growth_rate, bn_size):
        super(_DenseLayer, self).__init__()
        self.add_module('norm1', nn.BatchNorm2d(in_channels))
        self.add_module('relu1', nn.ReLU(inplace=True))
        self.add_module('conv1', nn.Conv2d(in_channels, bn_size * growth_rate,
                                           kernel_size=1,
                                           stride=1, bias=False))
        self.add_module('norm2', nn.BatchNorm2d(bn_size*growth_rate))
        self.add_module('relu2', nn.ReLU(inplace=True))
        self.add_module('conv2', nn.Conv2d(bn_size*growth_rate, growth_rate,
                                           kernel_size=3,
                                           stride=1, padding=1, bias=False))

    def forward(self, x):
        new_features = super(_DenseLayer, self).forward(x)
        return torch.cat([x, new_features], 1)


class _DenseBlock(nn.Sequential):
    def __init__(self, num_layers, in_channels, bn_size, growth_rate):
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            self.add_module('denselayer%d' % (i+1),
                            _DenseLayer(in_channels+growth_rate*i,
                                        growth_rate, bn_size))


class _Transition(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super(_Transition, self).__init__()
        self.add_module('norm', nn.BatchNorm2d(in_channels))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', nn.Conv2d(in_channels, out_channels,
                                          kernel_size=1,
                                          stride=1, bias=False))
        self.add_module('pool', nn.AvgPool2d(kernel_size=2, stride=2))


class DenseNet_BC(nn.Module):
    def __init__(self, growth_rate=12, block_config=(6,12,24,16),
                 bn_size=4, theta=0.5):
        super(DenseNet_BC, self).__init__()

        num_init_feature = 2 * growth_rate


        self.features = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv2d(3, num_init_feature,
                                kernel_size=7, stride=2,
                                padding=3, bias=False)),
            ('norm0', nn.BatchNorm2d(num_init_feature)),
            ('relu0', nn.ReLU(inplace=True)),
            ('pool0', nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        ]))



        num_feature = num_init_feature
        for i, num_layers in enumerate(block_config):
            self.features.add_module('denseblock%d' % (i+1),
                                     _DenseBlock(num_layers, num_feature,
                                                 bn_size, growth_rate))
            num_feature = num_feature + growth_rate * num_layers
            if i != len(block_config)-1 and i != 2:
                self.features.add_module('transition%d' % (i + 1),
                                         _Transition(num_feature,
                                                     int(num_feature * theta)))
                num_feature = int(num_feature * theta)

        self.features.add_module('norm5', nn.BatchNorm2d(num_feature))
        self.features.add_module('relu5', nn.ReLU(inplace=True))


    def init_weights(self, pretrained=None):
        if isinstance(pretrained, str):
            load_checkpoint(self, pretrained, strict=False)
        elif pretrained is None:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    kaiming_init(m)
                elif isinstance(m, nn.BatchNorm2d):
                    constant_init(m, 1)

    def forward(self, x):
        x = self.features.conv0(x)
        x = self.features.norm0(x)
        x = self.features.relu0(x)
        x = self.features.pool0(x)

        x1 = self.features.denseblock1(x)
        print("x1 ****************")
        x2 = self.features.transition1(x1)
        x2 = self.features.denseblock2(x2)
        print("x2 ****************")
        x3 = self.features.transition2(x2)
        x3 = self.features.denseblock3(x3)
        print("x3 ****************")
        x4 = self.features.denseblock4(x3)
        print("x4 ****************")
        x5 = self.features.transition4(x4)
        x5 = self.features.denseblock5(x5)
        print("x5 ****************")
        # print("x1 :", x1.shape)
        # print("x2 :", x2.shape)
        # print("x3 :", x3.shape)
        # print("x4 :", x4.shape)
        # print("x5 :", x5.shape)
        return tuple([x1,x2,x3,x4,x5])


def densenet(depth=121, pretrained=None):

    growth_rate = 32
    if depth == 121:
        block_config = (6, 12, 12, 12, 16)
    elif depth == 169:
        block_config = (6, 12, 32, 32)
    elif depth == 201:
        block_config = (6, 12, 24, 24, 32)
    elif depth == 264:
        block_config = (6, 12, 64, 48)
    else:
        raise ValueError("no supported depth")

    model = DenseNet_BC(growth_rate=growth_rate, block_config=block_config)
    #print(model)
    model.init_weights(pretrained)

    return model

