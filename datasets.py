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
        self.annotation_dir = '/home/mstveras/ssd-360/dataset/train/labels_all'
        
        # Load all image files, sorting them to ensure that they are aligned
        self.image_filenames = [os.path.join(self.image_dir, f) for f in sorted(os.listdir(self.image_dir)) if f.endswith('.jpg')][:max_images]
        self.annotation_filenames = [os.path.join(self.annotation_dir, f) for f in sorted(os.listdir(self.annotation_dir)) if f.endswith('.xml')][:max_images]
        
        assert len(self.image_filenames) == len(self.annotation_filenames)

        for img_filename, ann_filename in zip(self.image_filenames, self.annotation_filenames):
            img_basename = os.path.splitext(img_filename)[0][-7:-3]
            ann_basename = os.path.splitext(ann_filename)[0][-7:-3]
            #print(img_basename, ann_basename)
            assert img_basename == ann_basename, f"File name mismatch: {img_filename} and {ann_filename}"

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

        label_mapping = {
        'airconditioner': 0,
        'backpack': 1,
        'bathtub': 2,
        'bed': 3,
        'board': 4,
        'book': 5,
        'bottle': 6,
        'bowl': 7,
        'cabinet': 8,
        'chair': 9,
        'clock': 10,
        'computer': 11,
        'cup': 12,
        'door': 13,
        'fan': 14,
        'fireplace': 15,
        'heater': 16,
        'keyboard': 17,
        'light': 18,
        'microwave': 19,
        'mirror': 20,
        'mouse': 21,
        'oven': 22,
        'person': 23,
        'phone': 24,
        'picture': 25,
        'potted plant': 26,
        'refrigerator': 27,
        'sink': 28,
        'sofa': 29,
        'table': 30,
        'toilet': 31,
        'tv': 32,
        'vase': 33,
        'washer': 34,
        'window': 35,
        'wine glass': 36}

        w, h = 1920, 960
        new_w, new_h = 600, 300


        # The coordinates for each bounding box are given in the format (θ, ϕ, α, β), where:
        # θ (theta) represents the longitudinal angle of the bounding box center. This is an angle that goes around the equator of the sphere, 
        # with 0° usually being the prime meridian, and it ranges from -180° to 180°.
        #
        # ϕ (phi) represents the latitudinal angle of the bounding box center. This is an angle that goes from the south to the north pole of the sphere, 
        # with 0° being the equator, and it ranges from -90° to 90°.
        #
        # α (alpha) is the horizontal field of view of the bounding box. This is the angular extent of the box measured in the plane parallel to the equator, 
        # indicating how wide the box is from left to right.
        #
        # β (beta) is the vertical field of view of the bounding box. This is the angular extent of the box measured in a plane perpendicular to the equator, 
        # indicating the height of the box from top to bottom.


        for obj in root.findall('object'):
            #if obj.find('name').text == 'light':  # Check if the object is a person
            if True:
                bbox = obj.find('bndbox')

                # Normalize pixel coordinates of center to [-1, 1]
                x_center = int(bbox.find('x_center').text)*(new_w/w)/(600)
                y_center = int(bbox.find('y_center').text)*(new_h/h)/(300) 
                width = (float(bbox.find('width').text))/90
                height = (int(bbox.find('height').text))/90

                boxes.append([x_center, y_center, width, height, 0])
                labels.append(label_mapping[obj.find('name').text])
                confidences.append(1)

        boxes = torch.FloatTensor(boxes)
        labels = torch.LongTensor(labels)
        confidences = torch.FloatTensor(confidences).unsqueeze(1)  # Convert to tensor
        
        #CHECAR ESSA PARTE!!!!!!
        
        image, labels, difficulties = transform(image, labels, difficulties, split=self.split, new_w = 600, new_h = 300) 


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
