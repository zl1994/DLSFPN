import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import argparse
import collections

import numpy as np

import torch
import torch.optim as optim
from torchvision import transforms

from retinanet import model
from retinanet.dataloader import CocoDataset, CSVDataset, collater, Resizer, AspectRatioBasedSampler, Augmenter, \
    Normalizer
from torch.utils.data import DataLoader

from retinanet import coco_eval
from retinanet import csv_eval

assert torch.__version__.split('.')[0] == '1'

print('CUDA available: {}'.format(torch.cuda.is_available()))



def main(args=None):
    parser = argparse.ArgumentParser(description='Simple test script for test a RetinaNet network.')
    parser.add_argument('--checkpoint', help='path of checkpoint, train from checkpoint', default="./model/retinanet_msdnet_block5_step7_early_exit_VOC/VOC_retinanet_1.pt")
    parser.add_argument('--dataset', help='Dataset type, must be one of csv or coco.', default="csv")
    parser.add_argument('--coco_path', help='Path to COCO directory', default="/mnt/C/voc")
    parser.add_argument('--csv_train', help='Path to file containing training annotations (see readme)', default="./data/VOC/train.csv")
    parser.add_argument('--csv_classes', help='Path to file containing class list (see readme)', default="./data/VOC/class_list.csv")
    parser.add_argument('--csv_val', help='Path to file containing validation annotations (optional, see readme)', default="./data/VOC/test.csv")

    parser = parser.parse_args(args)

    if parser.dataset == 'coco':

        if parser.coco_path is None:
            raise ValueError('Must provide --coco_path when training on COCO,')

        dataset_val = CocoDataset(parser.coco_path, set_name='val2017',
                                  transform=transforms.Compose([Normalizer(), Resizer()]))

    elif parser.dataset == 'csv':

        if parser.csv_classes is None:
            raise ValueError('Must provide --csv_classes when training on COCO,')
        if parser.csv_val is None:
            dataset_val = None
            print('No validation annotations provided.')
        else:
            dataset_val = CSVDataset(train_file=parser.csv_val, class_list=parser.csv_classes,
                                     transform=transforms.Compose([Normalizer(), Resizer()]))

    else:
        raise ValueError('Dataset type not understood (must be csv or coco), exiting.')

    retinanet = torch.load(parser.checkpoint)

    use_gpu = True

    if use_gpu:
        if torch.cuda.is_available():
            retinanet = retinanet.cuda()

    if parser.dataset == 'coco':

        print('Evaluating dataset')

        coco_eval.evaluate_coco(dataset_val, retinanet)

    elif parser.dataset == 'csv' and parser.csv_val is not None:

        print('Evaluating dataset' + "   csv")

        mAP, mAP_res = csv_eval.evaluate(dataset_val, retinanet, early_exit=retinanet.early_exit)

        print("map:", mAP)


if __name__=="__main__":
    main()