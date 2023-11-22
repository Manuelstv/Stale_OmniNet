import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import xml.etree.ElementTree as ET
from utils import transform
import cv2
import numpy as np

class PascalVOCDataset(Dataset):

    def __init__(self, split, keep_difficult=False, max_images=10):
        self.split = split.upper()
        assert self.split in {'TRAIN', 'TEST', 'VAL'}
        #self.data_folder = data_folder
        self.keep_difficult = keep_difficult
        #self.split_dir = os.path.join(data_folder, self.split.lower())
        self.image_dir = '/home/mstveras/ssd-360/dataset/train/images'
        self.annotation_dir = '/home/mstveras/ssd-360/dataset/train/labels'
        
        # Load all image files, sorting them to ensure that they are aligned
        self.image_filenames = [os.path.join(self.image_dir, f) for f in sorted(os.listdir(self.image_dir)) if f.endswith('.jpg')][:max_images]
        self.annotation_filenames = [os.path.join(self.annotation_dir, f) for f in sorted(os.listdir(self.annotation_dir)) if f.endswith('.xml')][:max_images]
        
        #assert len(self.image_filenames) == len(self.annotation_filenames)

        #for img_filename, ann_filename in zip(self.image_filenames, self.annotation_filenames):
        #    img_basename = os.path.splitext(img_filename)[0][-7:-3]
        #    ann_basename = os.path.splitext(ann_filename)[0][-7:-3]
            #print(img_basename, ann_basename)
        #    assert img_basename == ann_basename, f"File name mismatch: {img_filename} and {ann_filename}"

        # If max_images is set, limit the dataset size
        if max_images is not None:
            self.image_filenames = self.image_filenames[:max_images]
            self.annotation_filenames = self.annotation_filenames[:max_images]

    def __getitem__(self, i):
        image_filename = self.image_filenames[i]
        annotation_filename = self.annotation_filenames[i]
        image = cv2.imread(image_filename)
        
        tree = ET.parse(annotation_filename)
        root = tree.getroot()
        boxes = []
        labels = []
        confidences = []
        difficulties = []

        label_mapping = {'nada':0, 'airconditioner': 1, 'backpack': 2, 'bathtub': 3, 'bed': 4, 'board': 5, 'book': 6, 'bottle': 7, 'bowl': 8, 'cabinet': 9, 'chair': 10, 'clock': 11, 'computer': 12, 'cup': 13, 'door': 14, 'fan': 15, 'fireplace': 16, 'heater': 17, 'keyboard': 18, 'light': 19, 'microwave': 20, 'mirror': 21, 'mouse': 22, 'oven': 23, 'person': 24, 'phone': 25, 'picture': 26, 'potted plant': 27, 'refrigerator': 28, 'sink': 29, 'sofa': 30, 'table': 31, 'toilet': 32, 'tv': 33, 'vase': 34, 'washer': 35, 'window': 36, 'wine glass': 37}

        for obj in root.findall('object'):
            bbox = obj.find('bndbox')
            x_center = int(bbox.find('x_center').text)/1920
            y_center = int(bbox.find('y_center').text)/960
            #phi = float(bbox.find('phi').text)
            #theta = float(bbox.find('theta').text)
            width = (float(bbox.find('width').text))/20#/40
            height = (float(bbox.find('height').text))/20#/40
            boxes.append([x_center, y_center, width, height])
            labels.append(label_mapping[obj.find('name').text])
            confidences.append(1)

        ##image = image.astype(np.float32) / 255.0
        #image = torch.from_numpy(image).permute(2, 0, 1)

        boxes = torch.FloatTensor(boxes)
        labels = torch.LongTensor(labels)
        confidences = torch.FloatTensor(confidences).unsqueeze(1)  # Convert to tensor
        image, boxes, labels, difficulties = transform(image, boxes, labels, difficulties, split=self.split, new_h = 300, new_w = 600) 


        return image, boxes, labels, confidences

    def __len__(self):
        return len(self.image_filenames)

    def collate_fn(self, batch):
        """
        Since each image may have a different number of objects, we need a collate function (to be passed to the DataLoader).

        This describes how to combine these tensors of different sizes. We use lists.

        Note: this need not be defined in this Class, can be standalone.

        :param batch: an iterable of N sets from __getitem__()
        :return: a tensor of images, lists of varying-size tensors of bounding boxes, labels, and difficulties
        """

        images = [item[0] for item in batch]
        boxes = [item[1] for item in batch]
        labels = [item[2] for item in batch]
        confidences = [item[3] for item in batch]

        images = torch.stack(images, dim=0)

        return images, boxes, labels, confidences  # tensor (N, 3, 300, 300), 3 lists of N tensors each
