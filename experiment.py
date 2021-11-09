import argparse

from torchvision import transforms

from retinanet.model import *
from retinanet.dataloader import CocoDataset, CSVDataset, collater, Resizer, AspectRatioBasedSampler, Augmenter, \
    Normalizer
import os

from op_counter import measure_model



os.environ["CUDA_VISIBLE_DEVICES"] = "0"




def main(args=None):
    parser = argparse.ArgumentParser(description='Simple training script for training a RetinaNet network.')
    parser.add_argument('--csv_train', help='Path to file containing training annotations (see readme)',
                        default="./data/VOC/train.csv")
    parser.add_argument('--csv_classes', help='Path to file containing class list (see readme)',
                        default="./data/VOC/class_list.csv")
    parser.add_argument('--csv_val', help='Path to file containing validation annotations (optional, see readme)',
                        default="./data/VOC/test.csv")

    parser.add_argument('--depth', help='Resnet depth, must be one of 18, 34, 50, 101, 152', type=int, default=50)
    parser.add_argument('--nBlocks', help='blocks of msdnet', type=int, default=5)
    parser.add_argument('--early_exit', help='early_exit', type=bool, default=True)

    parser = parser.parse_args(args)
    dataset_train = CSVDataset(train_file=parser.csv_train, class_list=parser.csv_classes,
                               transform=transforms.Compose([Normalizer(), Augmenter(), Resizer()]))

    retinanet = Retinanet(backbone='msdnet', depth=parser.depth, block=parser.nBlocks, step=7, num_classes=dataset_train.num_classes(),
                                    pretrained='./MSDNet_ImageNet/step=7/msdnet-step=7-block=5.pth.tar', early_exit=parser.early_exit)

    retinanet.eval()
    n_flops, n_params = measure_model(retinanet, 600, 1000)
    print(n_flops, n_params)
    del (retinanet)


if __name__=="__main__":
    main()

