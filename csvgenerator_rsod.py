import os
import csv
import xml.etree.ElementTree as ET

dataset_split = ["VOC2007", "VOC2012"]
root = "/mnt/C/RSOD/aircraft"



def train_generate(root):
    count  =0
    train = []
    path = '/mnt/C/RSOD/aircraft/Annotation/xml/'
    file_list = os.listdir(path)
    for file in file_list[:-100]:
        count+=1
        print(count)
        name = file.split('.')[0]
        img_path = root + "/JPEGImages/"+name+'.jpg'
        xml_path = root + "/Annotation/xml/"+name+'.xml'
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

    with open('./data/RSOD/train.csv', 'w') as csvfile:
        writer = csv.writer(csvfile)
        for row in train:
            writer.writerow(row)

def test_generate(root):
    count  =0
    train = []
    path = '/mnt/C/RSOD/aircraft/Annotation/xml/'
    file_list = os.listdir(path)
    for file in file_list[-100:]:
        count+=1
        print(count)
        name = file.split('.')[0]
        img_path = root + "/JPEGImages/"+name+'.jpg'
        xml_path = root + "/Annotation/xml/"+name+'.xml'
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

    with open('./data/RSOD/test.csv', 'w') as csvfile:
        writer = csv.writer(csvfile)
        for row in train:
            writer.writerow(row)


def class_list_generate():
    cla = []
    CLASSES = ('aircraft')
    cla.append(['aircraft', 0])
    '''
    for i , e in enumerate(CLASSES):
        cla.append([e, i])
    '''
    with open('./data/RSOD/class_list.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        for row in cla:
            writer.writerow(row)
"""
def set_generate():
    cla = []
    path = '/mnt/C/RSOD/aircraft/Annotation/xml/'
    file_list = os.listdir(path)
    for file in file_list:
        name = 
    cla.append(['aircraft', 0])
    '''
    for i , e in enumerate(CLASSES):
        cla.append([e, i])
    '''
    with open('./data/RSOD/class_list.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        for row in cla:
            writer.writerow(row)
"""
if __name__=="__main__":
    #train_generate(root)
    test_generate(root)
    #class_list_generate()
