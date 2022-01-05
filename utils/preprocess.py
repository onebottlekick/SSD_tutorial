import os
from urllib import request
import tarfile
import zipfile
from xml.etree import ElementTree
import numpy as np


def downloader(kind='data', root='../data'):
    os.makedirs(root, exist_ok=True)
    if kind == 'data':
        url = 'http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar'
    elif kind == 'vgg16_pth':
        url = 'https://s3.amazonaws.com/amdegroot-models/vgg16_reducedfc.pth'
    elif kind == 'ssd300_pth':
        url = 'https://s3.amazonaws.com/amdegroot-models/ssd300_mAP_77.43_v2.pth'
        
    file_path = os.path.join(root, url.split('/')[-1])    
    file_type = file_path.split('.')[-1]
    
    if os.path.exists(file_path):
        print('file already exists.')
        return
    
    print('downloading...')
    request.urlretrieve(url, file_path)
    print('done')
    
    print('extracting...')
    if file_type == 'tar':
        with tarfile.TarFile(file_path) as tar_file:
            tar_file.extractall(root)
        
    elif file_type == 'zip':
        with zipfile.ZipFile(file_path) as zip_file:
            zip_file.extractall(root)
    print('done')
    
if __name__ == '__main__':
    downloader()
    

def make_data_path_list(root='../data/VOCdevkit/VOC2012/'):
    train_id_names = os.path.join(root, 'ImageSets/Main/train.txt')
    val_id_names = os.path.join(root, 'ImageSets/Main/val.txt')
    
    train_img_list = []
    train_annotation_list = []    
    with open(train_id_names) as train:
        train_ids = train.readlines()
        for train_id in train_ids:
            train_img_list.append(os.path.join(root, 'JPEGImages', f'{train_id.strip()}.jpg'))
            train_annotation_list.append(os.path.join(root, 'Annotations', f'{train_id.strip()}.xml'))

    val_img_list = []
    val_annotation_list = []    
    with open(val_id_names) as val:
        val_ids = val.readlines()
        for val_id in val_ids:
            val_img_list.append(os.path.join(root, 'JPEGImages', f'{val_id.strip()}.jpg'))
            val_annotation_list.append(os.path.join(root, 'Annotations', f'{val_id.strip()}.xml'))
        
    return train_img_list, train_annotation_list, val_img_list, val_annotation_list


class XML2List:
    def __init__(self, classes):
        self.classes = classes
        
    def __call__(self, xml_path, width, height):
        result = []
        xml = ElementTree.parse(xml_path).getroot()
        
        for obj in xml.iter('object'):
            difficult = int(obj.find('difficult').text)
            if difficult == 1:
                continue
            
            bndbox = []
            
            name = obj.find('name').text.lower().strip()
            bbox = obj.find('bndbox')

            pts = ['xmin', 'ymin', 'xmax', 'ymax']

            for pt in pts:
                cur_pxl = int(bbox.find(pt).text) - 1
                
                if pt == 'xmin' or pt == 'xmax':
                    cur_pxl /= width
                else:
                    cur_pxl /= height
                    
                bndbox.append(cur_pxl)

            label_idx = self.classes.index(name)
            bndbox.append(label_idx)

            result += [bndbox]

        return np.array(result)