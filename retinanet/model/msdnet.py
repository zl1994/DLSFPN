import torch.nn as nn
import torch
import math
from mmcv.runner import load_checkpoint
import logging
from mmcv.utils import get_logger

# c3 c4 c5
feature_extract_layer = [ [4, 7, 10, 14, 18], [5, 10, 15, 21, 27], [7, 14, 21, 28, 35] ]
# fpn_sizes = [ [192, 448, 576], [288, 768, 640], [384, 704, 608], [320, 608, 1056], [448, 992, 976] ]

def get_root_logger(log_file=None, log_level=logging.INFO):
    logger = get_logger(name='mmdet', log_file=log_file, log_level=log_level)

    return logger

class ConvBasic(nn.Module):
    def __init__(self, nIn, nOut, kernel=3, stride=1,
                 padding=1):
        super(ConvBasic, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(nIn, nOut, kernel_size=kernel, stride=stride,
                      padding=padding, bias=False),
            nn.BatchNorm2d(nOut),
            nn.ReLU(True)
        )

    def forward(self, x):
        return self.net(x)

class ConvBN(nn.Module):
    def __init__(self, nIn, nOut, type: str, bottleneck,
                 bnWidth):
        """
        a basic conv in MSDNet, two type
        :param nIn:
        :param nOut:
        :param type: normal or down
        :param bottleneck: use bottlenet or not
        :param bnWidth: bottleneck factor
        """
        super(ConvBN, self).__init__()
        layer = []
        nInner = nIn
        if bottleneck is True:
            nInner = min(nInner, bnWidth * nOut)
            layer.append(nn.Conv2d(
                nIn, nInner, kernel_size=1, stride=1, padding=0, bias=False))
            layer.append(nn.BatchNorm2d(nInner))
            layer.append(nn.ReLU(True))

        if type == 'normal':
            layer.append(nn.Conv2d(nInner, nOut, kernel_size=3,
                                   stride=1, padding=1, bias=False))
        elif type == 'down':
            layer.append(nn.Conv2d(nInner, nOut, kernel_size=3,
                                   stride=2, padding=1, bias=False))
        else:
            raise ValueError

        layer.append(nn.BatchNorm2d(nOut))
        layer.append(nn.ReLU(True))

        self.net = nn.Sequential(*layer)

    def forward(self, x):

        return self.net(x)

class ConvDownNormal(nn.Module):
    def __init__(self, nIn1, nIn2, nOut, bottleneck, bnWidth1, bnWidth2):
        super(ConvDownNormal, self).__init__()
        self.conv_down = ConvBN(nIn1, nOut // 2, 'down',
                                bottleneck, bnWidth1)
        self.conv_normal = ConvBN(nIn2, nOut // 2, 'normal',
                                  bottleneck, bnWidth2)

    def forward(self, x):
        res = [x[1],
               self.conv_down(x[0]),
               self.conv_normal(x[1])]
        return torch.cat(res, dim=1)

class ConvNormal(nn.Module):
    def __init__(self, nIn, nOut, bottleneck, bnWidth):
        super(ConvNormal, self).__init__()
        self.conv_normal = ConvBN(nIn, nOut, 'normal',
                                  bottleneck, bnWidth)

    def forward(self, x):
        if not isinstance(x, list):
            x = [x]
        res = [x[0],
               self.conv_normal(x[0])]

        return torch.cat(res, dim=1)

class ParallelModule(nn.Module):
    """
    This module is similar to luatorch's Parallel Table
    input: N tensor
    network: N module
    output: N tensor
    """

    def __init__(self, parallel_modules):
        super(ParallelModule, self).__init__()
        self.m = nn.ModuleList(parallel_modules)

    def forward(self, x):
        x, feature = x[0], x[1]
        res = []
        for i in range(len(x)):
            res.append(self.m[i](x[i]))

        return tuple([res, feature])

class MSDNFirstLayer(nn.Module):
    def __init__(self, nIn, nOut, grFactor, nScales ):
        # nIn  3   nOut   32
        super(MSDNFirstLayer, self).__init__()    # grFactor [1,2,4, 4]
        self.layers = nn.ModuleList()

        conv = nn.Sequential(
            nn.Conv2d(nIn, nOut * grFactor[0], 7, 2, 3),
            nn.BatchNorm2d(nOut * grFactor[0]),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2, 1))
        self.layers.append(conv)

        nIn = nOut * grFactor[0]

        for i in range(1, nScales):
            self.layers.append(ConvBasic(nIn, nOut * grFactor[i],
                                         kernel=3, stride=2, padding=1))
            nIn = nOut * grFactor[i]

    def forward(self, x):
        x , feature = x[0], x[1]
        res = []
        for i in range(len(self.layers)):
            x = self.layers[i](x)
            res.append(x)

        return tuple([res, feature])

class MSDNLayer(nn.Module):
    def __init__(self, nIn, nOut, nScales, grFactor, bnFactor, bottleneck, inScales=None, outScales=None, n_layer_curr=-1):
        super(MSDNLayer, self).__init__()
        self.nIn = nIn
        self.nOut = nOut
        self.inScales = inScales if inScales is not None else nScales
        self.outScales = outScales if outScales is not None else nScales

        self.nScales = nScales
        self.discard = self.inScales - self.outScales

        self.offset = self.nScales - self.outScales
        self.layers = nn.ModuleList()
        self.n_layer_curr = n_layer_curr

        if self.discard > 0:
            nIn1 = nIn * grFactor[self.offset - 1]
            nIn2 = nIn * grFactor[self.offset]
            _nOut = nOut * grFactor[self.offset]
            self.layers.append(ConvDownNormal(nIn1, nIn2, _nOut, bottleneck,
                                              bnFactor[self.offset - 1],
                                              bnFactor[self.offset]))
        else:
            self.layers.append(ConvNormal(nIn * grFactor[self.offset],
                                          nOut * grFactor[self.offset],
                                          bottleneck,
                                          bnFactor[self.offset]))

        for i in range(self.offset + 1, self.nScales):
            nIn1 = nIn * grFactor[i - 1]
            nIn2 = nIn * grFactor[i]
            _nOut = nOut * grFactor[i]
            self.layers.append(ConvDownNormal(nIn1, nIn2, _nOut, bottleneck,
                                              bnFactor[i - 1],
                                              bnFactor[i]))

    def forward(self, x):
        x, feature = x[0], x[1]
        if self.discard > 0:
            inp = []
            for i in range(1, self.outScales + 1):
                inp.append([x[i - 1], x[i]])
        else:
            inp = [[x[0]]]
            for i in range(1, self.outScales):
                inp.append([x[i - 1], x[i]])

        res = []
        for i in range(self.outScales):
            res.append(self.layers[i](inp[i]))

        for i in range(len(feature_extract_layer)):
            if self.n_layer_curr in feature_extract_layer[i]:
                feature[feature_extract_layer[i].index(self.n_layer_curr)].append(res[i-3])

        return tuple([res, feature])

class MSDNet(nn.Module):
    def __init__(self, nBlocks = 5, in_channels=3, stem_channels = 32, base = 4, step=4, stepmode = "even",
                 growthRate=16, grFactor="1-2-4-4", bnFactor="1-2-4-4", prune="max", reduction=0.5, bottleneck = True):
        super(MSDNet, self).__init__()
        self.blocks = nn.ModuleList()
        self.classifier = nn.ModuleList()
        self.nBlocks = nBlocks  # nBlocks = 5 on ImageNet
        self.in_channels = in_channels
        base = step
        self.steps = [base]  # default: step 4, stepmode even, base 4
        self.grFactor = list(map(int, grFactor.split('-')))
        self.bnFactor = list(map(int, bnFactor.split('-')))
        self.nScales = len(self.grFactor)
        self.prune = prune
        self.reduction = reduction
        self.bottleneck = bottleneck
        self.growthRate = growthRate

        n_layers_all, n_layer_curr = base, 0
        for i in range(1, self.nBlocks):
            self.steps.append(step if stepmode == 'even'  # steps  [4, 4, 4, 4, 4]
                              else step * i + 1)
            n_layers_all += self.steps[-1]  # n_layers_all  20
            # n_layers_cur  0
        print("building network of steps: ")
        print(self.steps, n_layers_all)

        nIn = stem_channels  # 32
        for i in range(self.nBlocks):
            print(' ********************** Block {} '
                  ' **********************'.format(i + 1))
            m, nIn = \
                self._build_block(nIn, self.steps[i],
                                  n_layers_all, n_layer_curr)
            self.blocks.append(m)
            n_layer_curr += self.steps[i]



    def init_weights(self, pretrained):
        if isinstance(pretrained, str):
            logger = get_root_logger()
            load_checkpoint(self, pretrained, strict=False, logger=logger)
        elif pretrained is None:
            for m in self.blocks:
                if hasattr(m, '__iter__'):
                    for _m in m:
                        self._init_weights(_m)
                else:
                    self._init_weights(m)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2. / n))
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            m.bias.data.zero_()

    def _build_block(self, nIn, step, n_layer_all, n_layer_curr):


        layers = [MSDNFirstLayer(self.in_channels, nIn, self.grFactor, self.nScales)] \
            if n_layer_curr == 0 else []

        for i in range(step):
            n_layer_curr += 1

            if self.prune == 'min':
                inScales = min(self.nScales, n_layer_all - n_layer_curr + 2)
                outScales = min(self.nScales, n_layer_all - n_layer_curr + 1)
            elif self.prune == 'max':
                interval = math.ceil(1.0 * n_layer_all / self.nScales)  # 5
                inScales = self.nScales - math.floor(1.0 * (max(0, n_layer_curr - 2)) / interval)
                outScales = self.nScales - math.floor(1.0 * (n_layer_curr - 1) / interval)
            else:
                raise ValueError

            layers.append(MSDNLayer(nIn, self.growthRate, self.nScales,
                                    self.grFactor, self.bnFactor, self.bottleneck, inScales, outScales, n_layer_curr))
            print('|\t\tinScales {} outScales {} inChannels {} outChannels {}\t\t|'.format(inScales, outScales, nIn,
                                                                                           self.growthRate))

            nIn += self.growthRate
            if self.prune == 'max' and inScales > outScales and \
                    self.reduction > 0:
                offset = self.nScales - outScales
                layers.append(
                    self._build_transition(nIn, math.floor(1.0 * self.reduction * nIn),
                                           outScales, offset))
                _t = nIn
                nIn = math.floor(1.0 * self.reduction * nIn)
                print('|\t\tTransition layer inserted! (max), inChannels {}, outChannels {}\t|'.format(_t, math.floor(
                    1.0 * self.reduction * _t)))
            elif self.prune == 'min' and self.reduction > 0 and \
                    ((n_layer_curr == math.floor(1.0 * n_layer_all / 3)) or
                     n_layer_curr == math.floor(2.0 * n_layer_all / 3)):
                offset = self.nScales - outScales
                layers.append(self._build_transition(nIn, math.floor(1.0 * self.reduction * nIn),
                                                     outScales, offset))

                nIn = math.floor(1.0 * self.reduction * nIn)
                print('|\t\tTransition layer inserted! (min)\t|')
            print("")

        return nn.Sequential(*layers), nIn

    def _build_transition(self, nIn, nOut, outScales, offset):
        net = []
        for i in range(outScales):
            net.append(ConvBasic(nIn * self.grFactor[offset + i],
                                 nOut * self.grFactor[offset + i],
                                 kernel=1, stride=1, padding=0))
        return ParallelModule(net)


    def forward(self, x):
        x = tuple([x, [[], [], [], [], []]])
        for i in range(self.nBlocks):
            x = self.blocks[i](x)

        feature = x[1]
       
        return feature

def msdnet(nBlocks, step, pretrained=None,  **kwargs):
    model = MSDNet(nBlocks=nBlocks, step=step, **kwargs)
    model.init_weights(pretrained)
    return model
