import torch
import torch.nn as nn

from mmcv.cnn import normal_init, bias_init_with_prob

#light head
class RegressionModel(nn.Module):
    def __init__(self, num_features_in, num_anchors=9, feature_size=256, shallow=False):
        super(RegressionModel, self).__init__()

        self.module = nn.ModuleList()
        if shallow:
            self.module.append(nn.Conv2d(num_features_in, feature_size, kernel_size=3, stride=1, padding=1))
            self.module.append(nn.ReLU())
        else:
            self.module.append(nn.Conv2d(num_features_in, feature_size, kernel_size=(1,5), stride=1, padding=(0,2)))
            self.module.append(nn.ReLU())
            self.module.append(nn.Conv2d(feature_size, feature_size, kernel_size=(5, 1), stride=1, padding=(2, 0)))
            self.module.append(nn.ReLU())
        self.output = nn.Conv2d(feature_size, num_anchors * 4, kernel_size=3, padding=1)

    def init_weight(self):
        for m in [self.output,]:
            normal_init(m, std=0.01)
        for m in self.module:
            if isinstance(m,nn.Conv2d):
                normal_init(m, std=0.01)


    def forward(self, x):

        for i in range(len(self.module)):
            x = self.module[i](x)

        out = self.output(x)

        # out is B x C x W x H, with C = 4*num_anchors
        out = out.permute(0, 2, 3, 1)

        return out.contiguous().view(out.shape[0], -1, 4)
class ClassificationModel(nn.Module):
    def __init__(self, num_features_in, num_anchors=9, num_classes=80, prior=0.01, feature_size=256, shallow=False):
        super(ClassificationModel, self).__init__()


        self.num_classes = num_classes
        self.num_anchors = num_anchors

        self.module = nn.ModuleList()
        if shallow:
            self.module.append(nn.Conv2d(num_features_in, feature_size, kernel_size=3, stride=1, padding=1))
            self.module.append(nn.ReLU())
        else:
            self.module.append(nn.Conv2d(num_features_in, feature_size, kernel_size=(1, 5), stride=1, padding=(0, 2)))
            self.module.append(nn.ReLU())
            self.module.append(nn.Conv2d(feature_size, feature_size, kernel_size=(5, 1), stride=1, padding=(2, 0)))
            self.module.append(nn.ReLU())
        self.output = nn.Conv2d(feature_size, num_anchors * num_classes, kernel_size=3, padding=1)
        self.output_act = nn.Sigmoid()

    def init_weight(self):
        for m in self.module:
            if isinstance(m, nn.Conv2d):
                normal_init(m, std=0.01)
        bias_cls = bias_init_with_prob(0.01)
        normal_init(self.output, std=0.01, bias=bias_cls)




    def forward(self, x):
        for i in range(len(self.module)):
            x = self.module[i](x)
        out = self.output(x)
        out = self.output_act(out)

        # out is B x C x W x H, with C = n_classes + n_anchors
        out1 = out.permute(0, 2, 3, 1)

        batch_size, width, height, channels = out1.shape

        out2 = out1.view(batch_size, width, height, self.num_anchors, self.num_classes)

        return out2.contiguous().view(x.shape[0], -1, self.num_classes)


#normal head

# class RegressionModel(nn.Module):
#     def __init__(self, num_features_in, num_anchors=9, feature_size=256):
#         super(RegressionModel, self).__init__()
#
#
#         self.conv1 = nn.Conv2d(num_features_in, feature_size, kernel_size=3, padding=1)
#         self.act1 = nn.ReLU()
#
#         self.conv2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
#         self.act2 = nn.ReLU()
#
#         self.conv3 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
#         self.act3 = nn.ReLU()
#
#         self.conv4 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
#         self.act4 = nn.ReLU()
#
#         self.output = nn.Conv2d(feature_size, num_anchors * 4, kernel_size=3, padding=1)
#
#     def init_weight(self):
#         for m in [self.conv1, self.conv2, self.conv3, self.conv4, self.output]:
#             normal_init(m, std=0.01)
#
#
#     def forward(self, x):
#
#         out = self.conv1(x)
#         out = self.act1(out)
#
#         out = self.conv2(out)
#         out = self.act2(out)
#
#         out = self.conv3(out)
#         out = self.act3(out)
#
#         out = self.conv4(out)
#         out = self.act4(out)
#
#         out = self.output(out)
#
#         # out is B x C x W x H, with C = 4*num_anchors
#         out = out.permute(0, 2, 3, 1)
#
#         return out.contiguous().view(out.shape[0], -1, 4)
#
#
# class ClassificationModel(nn.Module):
#     def __init__(self, num_features_in, num_anchors=9, num_classes=80, prior=0.01, feature_size=256):
#         super(ClassificationModel, self).__init__()
#
#
#         self.num_classes = num_classes
#         self.num_anchors = num_anchors
#
#         self.conv1 = nn.Conv2d(num_features_in, feature_size, kernel_size=3, padding=1)
#         self.act1 = nn.ReLU()
#
#         self.conv2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
#         self.act2 = nn.ReLU()
#
#         self.conv3 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
#         self.act3 = nn.ReLU()
#
#         self.conv4 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
#         self.act4 = nn.ReLU()
#
#         self.output = nn.Conv2d(feature_size, num_anchors * num_classes, kernel_size=3, padding=1)
#         self.output_act = nn.Sigmoid()
#
#     def init_weight(self):
#         for m in [self.conv1,self.conv2,self.conv3, self.conv4]:
#             normal_init(m, std=0.01)
#         bias_cls = bias_init_with_prob(0.01)
#         normal_init(self.output, std=0.01, bias=bias_cls)
#
#
#
#
#     def forward(self, x):
#
#         out = self.conv1(x)
#         out = self.act1(out)
#
#         out = self.conv2(out)
#         out = self.act2(out)
#
#         out = self.conv3(out)
#         out = self.act3(out)
#
#         out = self.conv4(out)
#         out = self.act4(out)
#
#         out = self.output(out)
#         out = self.output_act(out)
#
#         # out is B x C x W x H, with C = n_classes + n_anchors
#         out1 = out.permute(0, 2, 3, 1)
#
#         batch_size, width, height, channels = out1.shape
#
#         out2 = out1.view(batch_size, width, height, self.num_anchors, self.num_classes)
#
#         return out2.contiguous().view(x.shape[0], -1, self.num_classes)