import os
import csv
import xml.etree.ElementTree as ET

dataset_split = ["VOC2007", "VOC2012"]
root = "/mnt/C/voc/VOCdevkit/"



def train_generate(root):
    count  =0
    train = []
    for dataset in dataset_split:
        path = root + dataset + "/ImageSets/Main/trainval.txt"
        f = open(path, 'r')
        for line in f.readlines():
            count+=1
            print(count)
            name = line.split('\n')[0]
            img_path = root + dataset + "/JPEGImages/"+name+'.jpg'
            xml_path = root + dataset + "/Annotations/"+name+'.xml'
            anno = ET.parse(xml_path).getroot()
            for obj in anno.iter('object'):
                name = obj.find('name').text.lower().strip()
                bbox = obj.find('bndbox')
                difficult = int(obj.find('difficult').text)
                if difficult:
                    continue
                x1 = int(bbox.find('xmin').text) - 1
                x2 = int(bbox.find('xmax').text) - 1
                y1 = int(bbox.find('ymin').text) - 1
                y2 = int(bbox.find('ymax').text) - 1
                if y1>=y2 or x1>=x2:
                    continue
                train.append([img_path, x1, y1, x2, y2, name])
        f.close()
    with open('./data/VOC/train.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        for row in train:
            writer.writerow(row)

def test_generate(root):
    path = root + 'VOC2007/ImageSets/Main/test.txt'
    f = open(path, 'r')
    test = []
    for line in f.readlines():
        name = line.split('\n')[0]
        img_path = root + 'VOC2007/JPEGImages/'+name+'.jpg'
        xml_path = root + 'VOC2007/Annotations/'+name+'.xml'
        anno = ET.parse(xml_path).getroot()
        for obj in anno.iter('object'):
            name = obj.find('name').text.lower().strip()
            bbox = obj.find('bndbox')
            difficult = int(obj.find('difficult').text)
            if difficult:
                continue
            x1 = int(bbox.find('xmin').text) - 1
            x2 = int(bbox.find('xmax').text) - 1
            y1 = int(bbox.find('ymin').text) - 1
            y2 = int(bbox.find('ymax').text) - 1
            if y1>=y2 or x1>=x2:
                print(x1, x2, y1, y2)
                print(img_path)
            test.append([img_path, x1, y1, x2, y2, name])
    f.close()
    with open('./data/VOC/test.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        for row in test:
            writer.writerow(row)

def class_list_generate():
    cla = []
    CLASSES = (
        'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat',
        'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant',
        'sheep', 'sofa', 'train', 'tvmonitor')
    for i , e in enumerate(CLASSES):
        cla.append([e, i])
    with open('./data/VOC/class_list.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        for row in cla:
            writer.writerow(row)

if __name__=="__main__":
    train_generate(root)
     #test_generate(root)
    # class_list_generate()
