import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss
import torch.nn.init as init

from model import SimpleObjectDetector, SimpleObjectDetectorMobile, SimpleObjectDetectorResnet
from datasets import PascalVOCDataset
torch.cuda.empty_cache()

import cv2
import json
from sphiou import Sph
from plot_tools import process_and_save_image
from foviou import fov_iou, deg2rad, angle2radian, fov_giou_loss
from sph2pob import sph_iou_aligned, fov_iou_aligned
import math

def fov_iou_batch(gt_boxes, pred_boxes):
    # Initialize a tensor to store IoU values
    ious = torch.zeros((len(gt_boxes), len(pred_boxes)))

    # Iterate over each ground truth and predicted box pair
    for i, Bg in enumerate(gt_boxes):
        for j, Bd in enumerate(pred_boxes):
            ious[i, j] = sph_iou_aligned(Bg.unsqueeze(0), Bd.unsqueeze(0))
    return ious

def deg_to_rad(degrees):
    return [math.radians(degree) for degree in degrees]

def hungarian_matching(gt_boxes_in, pred_boxes_in):
    # Compute the batch IoUs
    pred_boxes = pred_boxes_in.clone()
    gt_boxes = gt_boxes_in.clone()

    gt_boxes[:, 0] = gt_boxes_in[:, 0]*360/2
    gt_boxes[:, 1] = gt_boxes_in[:, 1]*180/2
    gt_boxes[:, 2] = gt_boxes_in[:, 2]*90
    gt_boxes[:, 3] = gt_boxes_in[:, 3]*90

    gt_boxes = gt_boxes.to(torch.int)

    pred_boxes[:, 0] = pred_boxes_in[:, 0]*360/2
    pred_boxes[:, 1] = pred_boxes_in[:, 1]*180/2
    pred_boxes[:, 2] = pred_boxes_in[:, 2]*90
    pred_boxes[:, 3] = pred_boxes_in[:, 3]*90

    pred_boxes = pred_boxes.to(torch.int)

    iou_matrix = fov_iou_batch(gt_boxes, pred_boxes)

    # Convert IoUs to cost
    cost_matrix = 1 - iou_matrix.detach().numpy()

    # Apply Hungarian matching
    gt_indices, pred_indices = linear_sum_assignment(cost_matrix)

    # Extract the matched pairs
    matched_pairs = [(gt_boxes[i], pred_boxes[j]) for i, j in zip(gt_indices, pred_indices)]

    return matched_pairs, iou_matrix[gt_indices, pred_indices]

torch.cuda.empty_cache()
# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
num_epochs = 500
learning_rate = 0.0001
batch_size = 10
num_classes = 37
max_images = 100


# Initialize dataset and dataloader
train_dataset = PascalVOCDataset(split='TRAIN', keep_difficult=False, max_images=max_images)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=train_dataset.collate_fn)

def init_weights(m):
    if isinstance(m, nn.Linear):
        # Choose the initialization method here (e.g., Xavier, He)
        init.xavier_uniform_(m.weight)
        if m.bias is not None:
            init.zeros_(m.bias)


model = SimpleObjectDetector(num_boxes=20, num_classes=num_classes).to(device)
model.fc1.apply(init_weights)
model.det_head.apply(init_weights)

optimizer = optim.Adam(model.parameters(), lr=learning_rate)

for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    total_matches = 0
    total_regression_loss = torch.tensor(0.0, device=device, requires_grad=True)

    for i, (images, boxes_list, labels_list, confidences_list) in enumerate(train_loader):
        images = images.to(device)
        optimizer.zero_grad()
        detection_preds = model(images)
        n = 0

        for boxes, labels, det_preds in zip(boxes_list, labels_list, detection_preds):
            
            boxes = boxes.to(device)
            det_preds = det_preds.to(device)          
            labels = labels.to(device)
            matches, matched_iou_scores = hungarian_matching(boxes, det_preds)

            #print(matched_iou_scores)
            regression_loss = (1 - matched_iou_scores).mean()

            total_regression_loss = total_regression_loss + regression_loss
            total_matches += len(matches)
            
            #salvando images de 10 em 10 épocas
            if epoch>0 and epoch%10==0:
            #if True:
                #pass
                img1 = images[n].permute(1,2,0).cpu().numpy()*255
                img = process_and_save_image(img1, boxes.cpu().detach().numpy(), (0,255,0), f'/home/mstveras/images/img{n}.png')
                img = process_and_save_image(img, det_preds.cpu().detach().numpy(), (255,0,0), f'/home/mstveras/images/img2_{n}.png')
            
                n+=1

        if total_matches > 0:
            avg_regression_loss = total_regression_loss / total_matches
        else:
            avg_regression_loss = total_regression_loss*0
            print('not matched')

        # Backward pass and optimize
        avg_regression_loss.backward()
        optimizer.step()

        total_loss += avg_regression_loss.item()

    avg_epoch_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch}/{num_epochs}: Loss: {avg_epoch_loss}")

model_file = "best.pth"
torch.save(model.state_dict(), model_file)
print(f"Model saved to {model_file} after Epoch {epoch + 1}")
print('Training completed.')