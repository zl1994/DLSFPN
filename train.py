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

model_name = "retinanet_msdnet_block5_step7_early_exit_w_shared_fpn_lighthead_11111"
dataset_name = "RSOD"
save_path = "./model/"+model_name+"_"+dataset_name
pkl_save = save_path + "/dets_pkl"
if not os.path.exists(save_path):
    os.mkdir(save_path)
if not os.path.exists(pkl_save):
    os.mkdir(pkl_save)
def adjust_lr(optim, epoch, lr, epoch_num):
    if epoch_num==16:
        if epoch>=11 and epoch<14:
            lr = lr *0.3
        if epoch == 14 or epoch == 15:
            lr = lr * 0.1
        elif epoch == 16:
            lr = lr * 0.01
        for param in optim.param_groups:
            param["lr"] = lr
    elif epoch_num == 12:
        if epoch>9 and epoch<=11:
            lr = lr *0.1
        if epoch==12:
            lr = lr*0.01
        for param in optim.param_groups:
            param["lr"] = lr
    return lr

def adjust_lr_for_warmup(optim, lr, iter_num):
    lr = lr*(1 -  (1 - iter_num/500)*(1-0.001))
    for param in optim.param_groups:
        param["lr"] = lr
    return lr



def main(args=None):
    parser = argparse.ArgumentParser(description='Simple training script for training a RetinaNet network.')
    parser.add_argument('--checkpoint', help='path of checkpoint, train from checkpoint', default=None)

    #parser.add_argument('--checkpoint', help='path of checkpoint, train from checkpoint', default=save_path +"/VOC_retinanet_9.pt")
    parser.add_argument('--coco_path', help='Path to COCO directory', default="/mnt/C/coco_xzh")
    parser.add_argument('--csv_train', help='Path to file containing training annotations (see readme)', default="./data/RSOD/train.csv")
    parser.add_argument('--csv_classes', help='Path to file containing class list (see readme)', default="./data/RSOD/class_list.csv")
    parser.add_argument('--csv_val', help='Path to file containing validation annotations (optional, see readme)', default="./data/RSOD/test.csv")

    parser.add_argument('--depth', help='Resnet depth, must be one of 18, 34, 50, 101, 152', type=int, default=50)
    parser.add_argument('--nBlocks', help='blocks of msdnet', type=int, default=5)
    parser.add_argument('--epochs', help='Number of epochs', type=int, default=16)
    parser.add_argument('--learning_rate', help='learning rate', type=float, default=0.03)
    parser.add_argument('--early_exit', help='early_exit', type=bool, default=True)
    parser = parser.parse_args(args)


    # Create the data loaders
    if dataset_name == 'RSOD':
        dataset = 'csv'
    elif dataset_name == 'COCO':
        dataset = 'coco'
    else:
        raise ValueError("not supported dataset")


    if dataset == 'coco':

        if parser.coco_path is None:
            raise ValueError('Must provide --coco_path when training on COCO,')

        dataset_train = CocoDataset(parser.coco_path, set_name='train2017',
                                    transform=transforms.Compose([Normalizer(), Augmenter(), Resizer()]))
        dataset_val = CocoDataset(parser.coco_path, set_name='val2017',
                                  transform=transforms.Compose([Normalizer(), Resizer()]))

    elif dataset == 'csv':

        if parser.csv_train is None:
            raise ValueError('Must provide --csv_train when training on COCO,')

        if parser.csv_classes is None:
            raise ValueError('Must provide --csv_classes when training on COCO,')

        dataset_train = CSVDataset(train_file=parser.csv_train, class_list=parser.csv_classes,
                                   transform=transforms.Compose([Normalizer(), Augmenter(), Resizer()]))

        if parser.csv_val is None:
            dataset_val = None
            print('No validation annotations provided.')
        else:
            dataset_val = CSVDataset(train_file=parser.csv_val, class_list=parser.csv_classes,
                                     transform=transforms.Compose([Normalizer(), Resizer()]))

    else:
        raise ValueError('Dataset type not understood (must be csv or coco), exiting.')

    sampler = AspectRatioBasedSampler(dataset_train, batch_size=9, drop_last=False)
    dataloader_train = DataLoader(dataset_train,     num_workers=4, collate_fn=collater, batch_sampler=sampler)

    learning_rate = parser.learning_rate * sampler.batch_size/16

    # if dataset_val is not None:
    #     sampler_val = AspectRatioBasedSampler(dataset_val, batch_size=10, drop_last=False)
    #     dataloader_val = DataLoader(dataset_val, num_workers=3, collate_fn=collater, batch_sampler=sampler_val)

    # Create the model
    epoch_start = 0
    if parser.checkpoint:
        retinanet = torch.load(parser.checkpoint)
        epoch_start = int(parser.checkpoint[parser.checkpoint.find(dataset_name+"_retinanet_")+11+len(dataset_name):-3])
    else:
        retinanet = model.Retinanet(backbone='msdnet', depth=parser.depth, block=parser.nBlocks, step=7, num_classes=dataset_train.num_classes(),
                                    pretrained='./MSDNet_ImageNet/step=7/msdnet-step=7-block=5.pth.tar', early_exit=parser.early_exit)

    use_gpu = True

    if use_gpu:
        if torch.cuda.is_available():
            retinanet = retinanet.cuda()

    if torch.cuda.is_available():
        retinanet = torch.nn.DataParallel(retinanet).cuda()
    else:
        retinanet = torch.nn.DataParallel(retinanet)

    retinanet.training = True

    #optimizer = optim.Adam(retinanet.parameters(), lr=1e-5)
    optimizer = optim.SGD(retinanet.parameters(), lr = learning_rate, momentum=0.9, weight_decay=0.0001)

    #scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, verbose=True)

    loss_hist = collections.deque(maxlen=500)

    retinanet.train()
    retinanet.module.freeze_bn()

    print('Num training images: {}'.format(len(dataset_train)))

    for epoch_num in range(epoch_start+1, parser.epochs+1):


        lr_now = adjust_lr(optimizer, epoch_num, learning_rate, parser.epochs)

        retinanet.train()
        retinanet.module.freeze_bn()

        epoch_loss = []

        for iter_num, data in enumerate(dataloader_train):
            try:
                if epoch_num==1 and iter_num<=500:
                    lr_now = adjust_lr_for_warmup(optimizer, learning_rate, iter_num)

                optimizer.zero_grad()

                if torch.cuda.is_available():
                    classification_loss, regression_loss = retinanet([data['img'].cuda().float(), data['annot']])
                else:
                    classification_loss, regression_loss = retinanet([data['img'].float(), data['annot']])



                loss = classification_loss + regression_loss

                if bool(loss == 0):
                    continue

                loss.backward()

                torch.nn.utils.clip_grad_norm_(retinanet.parameters(), 2)

                optimizer.step()

                loss_hist.append(float(loss))

                epoch_loss.append(float(loss))

                print(
                    'Epoch: {} | Iteration: {} | Classification loss: {:1.5f} | Regression loss: {:1.5f} | Running loss: {:1.5f} | learning rate: {:1.5f}'.format(
                        epoch_num, iter_num, float(classification_loss), float(regression_loss), np.mean(loss_hist), lr_now))

                del classification_loss
                del regression_loss
            except Exception as e:
                print(e)
                continue


        torch.save(retinanet.module, save_path + '/{}_retinanet_{}.pt'.format(dataset_name, epoch_num))

        if epoch_num>=16:
            if dataset == 'coco':

                print('Evaluating dataset')

                coco_eval.evaluate_coco(dataset_val, retinanet, save_path=pkl_save + '/{}_retinanet_{}_dets.pkl'.format(dataset_name, epoch_num))

            elif dataset == 'csv' and parser.csv_val is not None:

                print('Evaluating dataset' + "   csv")

                csv_eval.evaluate(dataset_val, retinanet, early_exit=parser.early_exit, save_path=pkl_save + '/{}_retinanet_{}_dets_'.format(dataset_name, epoch_num))




    retinanet.eval()



if __name__ == '__main__':
    main()
