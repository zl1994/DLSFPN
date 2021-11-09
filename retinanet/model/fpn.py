import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import  xavier_init

class FeatureTransfer(nn.Module):
    def __init__(self, C1_size, C2_size, C3_size, C4_size, C5_size, feature_size=256):
        super(FeatureTransfer, self).__init__()
        self.head1_conv1 = nn.Conv2d(C1_size, feature_size, kernel_size=3, stride=2, padding=1)
        self.head1_bn1 = nn.BatchNorm2d(feature_size)
        self.head1_relu1 = nn.ReLU(inplace=True)
        self.head1_conv2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=2, padding=1)
        self.head1_bn2 = nn.BatchNorm2d(feature_size)
        self.head1_relu2 = nn.ReLU(inplace=True)
        self.head1_conv3 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=2, padding=1)
        self.head1_bn3 = nn.BatchNorm2d(feature_size)
        self.head1_relu3 = nn.ReLU(inplace=True)

        self.head2_conv1 = nn.Conv2d(C2_size, feature_size, kernel_size=3, stride=2, padding=1)
        self.head2_bn1 = nn.BatchNorm2d(feature_size)
        self.head2_relu1 = nn.ReLU(inplace=True)
        self.head2_conv2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=2, padding=1)
        self.head2_bn2 = nn.BatchNorm2d(feature_size)
        self.head2_relu2 = nn.ReLU(inplace=True)

        self.head3_conv1 = nn.Conv2d(C3_size, feature_size, kernel_size=3, stride=2, padding=1)
        self.head3_bn1 = nn.BatchNorm2d(feature_size)
        self.head3_relu1 = nn.ReLU(inplace=True)

        self.head4_conv1 = nn.Conv2d(C4_size, feature_size, kernel_size=3, stride=2, padding=1)
        self.head4_bn1 = nn.BatchNorm2d(feature_size)
        self.head4_relu1 = nn.ReLU(inplace=True)

    def init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                xavier_init(m, distribution='uniform')

    def forward(self, inputs):

        out = [[],[],[],[],[]]
        x1,x2,x3,x4,x5 = inputs

        print("feature process start ****************")
        head1_feature = self.head1_conv1(x1)
        head1_feature = self.head1_bn1(head1_feature)
        head1_feature = self.head1_relu1(head1_feature)
        out[0].append(head1_feature)
        head1_feature = self.head1_conv2(head1_feature)
        head1_feature = self.head1_bn2(head1_feature)
        head1_feature = self.head1_relu2(head1_feature)
        out[0].append(head1_feature)
        head1_feature = self.head1_conv3(head1_feature)
        head1_feature = self.head1_bn3(head1_feature)
        head1_feature = self.head1_relu3(head1_feature)
        out[0].append(head1_feature)
        print("feature process head1 end ****************")
        out[1].append(x2)
        head2_feature = self.head2_conv1(x2)
        head2_feature = self.head2_bn1(head2_feature)
        head2_feature = self.head2_relu1(head2_feature)
        out[1].append(head2_feature)
        head2_feature = self.head2_conv2(head2_feature)
        head2_feature = self.head2_bn2(head2_feature)
        head2_feature = self.head2_relu2(head2_feature)
        out[1].append(head2_feature)
        print("feature process head2 end ****************")
        out[2].append(x2)
        out[2].append(x3)
        head3_feature = self.head3_conv1(x3)
        head3_feature = self.head3_bn1(head3_feature)
        head3_feature = self.head3_relu1(head3_feature)
        out[2].append(head3_feature)
        print("feature process head3 end ****************")
        out[3].append(x2)
        out[3].append(x4)
        head4_feature = self.head4_conv1(x4)
        head4_feature = self.head4_bn1(head4_feature)
        head4_feature = self.head4_relu1(head4_feature)
        out[3].append(head4_feature)
        print("feature process head4 end ****************")
        out[4].append(x2)
        out[4].append(x4)
        out[4].append(x5)
        print("feature process head5 end ****************")
        return out

class PyramidFeatures(nn.Module):
    def __init__(self, C3_size, C4_size, C5_size, feature_size=256):
        super(PyramidFeatures, self).__init__()

        # upsample C5 to get P5 from the FPN paper
        self.P5_1 = nn.Conv2d(C5_size, feature_size, kernel_size=1, stride=1, padding=0)
        #self.P5_upsampled = nn.Upsample(scale_factor=2, mode='nearest')
        self.P5_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)

        # add P5 elementwise to C4
        self.P4_1 = nn.Conv2d(C4_size, feature_size, kernel_size=1, stride=1, padding=0)
        #self.P4_upsampled = nn.Upsample(scale_factor=2, mode='nearest')
        self.P4_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)

        # add P4 elementwise to C3
        self.P3_1 = nn.Conv2d(C3_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.P3_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)

        # "P6 is obtained via a 3x3 stride-2 conv on C5"
        self.P6 = nn.Conv2d(C5_size, feature_size, kernel_size=3, stride=2, padding=1)

        # "P7 is computed by applying ReLU followed by a 3x3 stride-2 conv on P6"
        self.P7_1 = nn.ReLU()
        self.P7_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=2, padding=1)

    def init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                xavier_init(m, distribution='uniform')


    def forward(self, inputs):
        C3, C4, C5 = inputs

        P5_x = self.P5_1(C5)
        #P5_upsampled_x = self.P5_upsampled(P5_x)
        P5_upsampled_x = F.interpolate(P5_x, size=C4.shape[2:], mode='nearest')
        P5_x = self.P5_2(P5_x)

        P4_x = self.P4_1(C4)
        P4_x = P5_upsampled_x + P4_x
        #P4_upsampled_x = self.P4_upsampled(P4_x)
        P4_upsampled_x = F.interpolate(P4_x, size=C3.shape[2:], mode='nearest')
        P4_x = self.P4_2(P4_x)

        P3_x = self.P3_1(C3)
        P3_x = P3_x + P4_upsampled_x
        P3_x = self.P3_2(P3_x)

        P6_x = self.P6(C5)

        #P7_x = self.P7_1(P6_x)
        P7_x = self.P7_2(P6_x)

        return [P3_x, P4_x, P5_x, P6_x, P7_x]

class SharedPyramidFeatures(nn.Module):
    def __init__(self, fpn_sizes, feature_size=256):
        super(SharedPyramidFeatures, self).__init__()

        self.P3_1 = nn.ModuleList()
        self.P4_1 = nn.ModuleList()
        self.P5_1 = nn.ModuleList()
        for i in range(len(fpn_sizes)):
            self.P3_1.append(nn.Conv2d(fpn_sizes[i][0], feature_size, kernel_size=1, stride=1, padding=0))
            self.P4_1.append(nn.Conv2d(fpn_sizes[i][1], feature_size, kernel_size=1, stride=1, padding=0))
            self.P5_1.append(nn.Conv2d(fpn_sizes[i][2], feature_size, kernel_size=1, stride=1, padding=0))
        self.P5_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)
        self.P4_2 = nn.Conv2d(feature_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.P3_2 = nn.Conv2d(feature_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.P6 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=2, padding=1)
        self.P7_1 = nn.ReLU()
        self.P7_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=2, padding=1)

    def init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                xavier_init(m, distribution='uniform')
    def forward(self, inputs, head_id):
        C3, C4, C5 = inputs

        P5_x = self.P5_1[head_id](C5)
        P6_x = self.P6(P5_x)

        P5_upsampled_x = F.interpolate(P5_x, size=C4.shape[2:], mode='nearest')
        P5_x = self.P5_2(P5_x)
        P4_x = self.P4_1[head_id](C4)
        P4_x = P5_upsampled_x + P4_x
        #P4_upsampled_x = self.P4_upsampled(P4_x)
        P4_upsampled_x = F.interpolate(P4_x, size=C3.shape[2:], mode='nearest')
        P4_x = self.P4_2(P4_x)
        P3_x = self.P3_1[head_id](C3)
        P3_x = P3_x + P4_upsampled_x
        P3_x = self.P3_2(P3_x)



        #P7_x = self.P7_1(P6_x)
        P7_x = self.P7_2(P6_x)

        return [P3_x, P4_x, P5_x, P6_x, P7_x]