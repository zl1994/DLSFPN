import torch
from torchvision.ops import nms
from retinanet.utils import  BBoxTransform, ClipBoxes
from retinanet.anchors import Anchors
from retinanet import losses
from .fpn import *
from .head import *
from .resnet import *
from .msdnet import *
from .densenet import *
import time
feature_size =  [64,64,96,96,128]
shallow = [True, True, False, False, False]
class Retinanet(nn.Module):
    def __init__(self, backbone, depth=None, block=None, step=None, num_classes=None, pretrained=None, early_exit=False):
        super(Retinanet, self).__init__()

        self.early_exit = early_exit

        if backbone == 'resnet':
            self.backbone = resnet(depth, pretrained)

            if self.early_exit:
                # fpn_sizes = [[288, 576, 576],[320, 640, 640],[448, 608, 608],[448, 992, 528],[448, 992, 976]]
                fpn_sizes = [[192, 448, 576], [288, 768, 640], [384, 704, 608], [320, 608, 1056], [448, 992, 976]]
                self.fpn = nn.ModuleList()
                self.regressionModel = nn.ModuleList()
                self.classificationModel = nn.ModuleList()
                for i in range(block):
                    self.fpn.append(PyramidFeatures(fpn_sizes[i][0], fpn_sizes[i][1], fpn_sizes[i][2]))
                    self.fpn[i].init_weight()
                    self.regressionModel.append(RegressionModel(256))
                    self.regressionModel[i].init_weight()
                    self.classificationModel.append(ClassificationModel(256, num_classes=num_classes))
                    self.classificationModel[i].init_weight()
            else:
                if depth < 50:
                    fpn_sizes = [128, 256, 512]
                else:
                    fpn_sizes = [512, 1024, 2048]
                self.fpn = PyramidFeatures(fpn_sizes[0], fpn_sizes[1], fpn_sizes[2])
                self.fpn.init_weight()

                self.regressionModel = RegressionModel(256)
                self.regressionModel.init_weight()

                self.classificationModel = ClassificationModel(256, num_classes=num_classes)
                self.classificationModel.init_weight()
        elif backbone == 'densenet':

            self.backbone = densenet(depth, pretrained)

            if self.early_exit:

                self.feature_transfer = FeatureTransfer(256, 512, 1024, 1792, 1920)
                self.feature_transfer.init_weight()

                fpn_sizes = [[256, 256, 256], [512, 256, 256], [512, 1024, 256], [512, 1792, 256], [512, 1792, 1920]]
                self.fpn = nn.ModuleList()
                self.regressionModel = nn.ModuleList()
                self.classificationModel = nn.ModuleList()
                for i in range(block):
                    self.fpn.append(PyramidFeatures(fpn_sizes[i][0], fpn_sizes[i][1], fpn_sizes[i][2]))
                    self.fpn[i].init_weight()
                    self.regressionModel.append(RegressionModel(256))
                    self.regressionModel[i].init_weight()
                    self.classificationModel.append(ClassificationModel(256, num_classes=num_classes))
                    self.classificationModel[i].init_weight()
            else:
                raise ValueError("not implemented yet!")

        elif backbone == 'msdnet':
            if block is None or step is None:
                raise ValueError('MSDNet need nblocks and step')
            self.backbone = msdnet(nBlocks=block, step=step, pretrained=pretrained)
            self.nBolcks = block
            if self.early_exit:
                # fpn_sizes = [[288, 576, 576],[320, 640, 640],[448, 608, 608],[448, 992, 528],[448, 992, 976]]
                fpn_sizes = [[192, 448, 576], [288, 768, 640], [384, 704, 608], [320, 608, 1056], [448, 992, 976]]
                self.fpn = SharedPyramidFeatures(fpn_sizes)
                self.fpn.init_weight()
                self.regressionModel = nn.ModuleList()
                self.classificationModel = nn.ModuleList()
                for i in range(block):
                    self.regressionModel.append(RegressionModel(num_features_in=256,feature_size=feature_size[i],shallow=shallow[i]))
                    self.regressionModel[i].init_weight()
                    self.classificationModel.append(ClassificationModel(num_features_in=256,feature_size=feature_size[i], num_classes=num_classes,shallow=shallow[i]))
                    self.classificationModel[i].init_weight()
            else:
                if block==5 and step==4:
                    fpn_sizes = [256, 544, 560]
                elif block==5 and step==7:
                    fpn_sizes = [448, 992, 976]
                self.fpn = PyramidFeatures(fpn_sizes[0], fpn_sizes[1], fpn_sizes[2])
                self.fpn.init_weight()

                self.regressionModel = RegressionModel(256)
                self.regressionModel.init_weight()

                self.classificationModel = ClassificationModel(256, num_classes=num_classes)
                self.classificationModel.init_weight()



        self.anchors = Anchors()

        self.regressBoxes = BBoxTransform()

        self.clipBoxes = ClipBoxes()

        self.focalLoss = losses.FocalLoss()

        self.freeze_bn()



    def freeze_bn(self):
        '''Freeze BatchNorm layers.'''
        for layer in self.modules():
            if isinstance(layer, nn.BatchNorm2d):
                layer.eval()


    def forward(self, inputs):

        if self.training:
            img_batch, annotations = inputs
        else:
            img_batch = inputs

        x = self.backbone(img_batch)

        classification_list, regression_list = [], []

        # anchor size [[1, level1, 4], [1, level2, 4], [1, level3, 4], [1, level4, 4], [1, level5, 4]]
        anchors = self.anchors(img_batch)

        if self.early_exit:
            for i in range(len(x)):
                print("fpn " + str(i + 1) + "  start************")
                features = self.fpn(x[i], i)
                print("fpn " + str(i + 1) + "  end************")
                print("head " + str(i + 1) + "  start************")
                regression_list.append(torch.cat([self.regressionModel[i](feature) for feature in features], dim=1))
                classification_list.append(torch.cat([self.classificationModel[i](feature) for feature in features], dim=1))
                print("head " + str(i + 1) + "  end************")
                #regression size [batch_size, num_anchor, 4] eg: [2, 94851, 4]

        else:
            x = x[-1]
            features = self.fpn(x)
            regression_list.append(torch.cat([self.regressionModel(feature) for feature in features], dim=1))
            classification_list.append(torch.cat([self.classificationModel(feature) for feature in features], dim=1))



        if self.training:
            loss_weights = [1, 1, 1, 1, 1]
            classification_loss, regression_loss = self.focalLoss(classification_list[0], regression_list[0], anchors, annotations)
            classification_loss, regression_loss = classification_loss*loss_weights[0], regression_loss*loss_weights[0]
            for i in range(1, len(classification_list)):
                classification_loss_t, regression_loss_t = self.focalLoss(classification_list[i], regression_list[i], anchors, annotations)
                classification_loss += classification_loss_t*loss_weights[i]
                regression_loss += regression_loss_t*loss_weights[i]
            return classification_loss/sum(loss_weights), regression_loss/sum(loss_weights)
        else:
            res = []
            for k in range(len(classification_list)):
                regression = regression_list[k]
                classification = classification_list[k]
                print(torch.max(classification.view(-1)))
                transformed_anchors = self.regressBoxes(anchors, regression)
                transformed_anchors = self.clipBoxes(transformed_anchors, img_batch)

                finalResult = [[], [], []]

                finalScores = torch.Tensor([])
                finalAnchorBoxesIndexes = torch.Tensor([]).long()
                finalAnchorBoxesCoordinates = torch.Tensor([])

                if torch.cuda.is_available():
                    finalScores = finalScores.cuda()
                    finalAnchorBoxesIndexes = finalAnchorBoxesIndexes.cuda()
                    finalAnchorBoxesCoordinates = finalAnchorBoxesCoordinates.cuda()

                for i in range(classification.shape[2]):
                    scores = torch.squeeze(classification[:, :, i])
                    scores_over_thresh = (scores > 0.05)
                    if scores_over_thresh.sum() == 0:


                        # no boxes to NMS, just continue
                        continue

                    scores = scores[scores_over_thresh]
                    anchorBoxes = torch.squeeze(transformed_anchors)
                    anchorBoxes = anchorBoxes[scores_over_thresh]
                    anchors_nms_idx = nms(anchorBoxes, scores, 0.5)

                    finalResult[0].extend(scores[anchors_nms_idx])
                    finalResult[1].extend(torch.tensor([i] * anchors_nms_idx.shape[0]))
                    finalResult[2].extend(anchorBoxes[anchors_nms_idx])

                    finalScores = torch.cat((finalScores, scores[anchors_nms_idx]))
                    finalAnchorBoxesIndexesValue = torch.tensor([i] * anchors_nms_idx.shape[0])
                    if torch.cuda.is_available():
                        finalAnchorBoxesIndexesValue = finalAnchorBoxesIndexesValue.cuda()

                    finalAnchorBoxesIndexes = torch.cat((finalAnchorBoxesIndexes, finalAnchorBoxesIndexesValue))
                    finalAnchorBoxesCoordinates = torch.cat((finalAnchorBoxesCoordinates, anchorBoxes[anchors_nms_idx]))

                res.append([finalScores, finalAnchorBoxesIndexes, finalAnchorBoxesCoordinates])

            return res









