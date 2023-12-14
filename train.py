import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
#sfrom torchvision import transforms
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss
import torch.nn.init as init

from model import SimpleObjectDetectorWithBackbone, SimpleObjectDetector
from datasets import PascalVOCDataset
torch.cuda.empty_cache()

import cv2
import json
from calculate_RoIoU import Sph
from plot_tools import process_and_save_image
from foviou import fov_iou, deg_to_rad

def fov_iou_batch(gt_boxes, pred_boxes):
    # Initialize a tensor to store IoU values
    ious = torch.zeros((len(gt_boxes), len(pred_boxes)))

    # Iterate over each ground truth and predicted box pair
    for i, Bg in enumerate(gt_boxes):
        for j, Bd in enumerate(pred_boxes):
            ious[i, j] = fov_iou(Bg, Bd)

    return ious

def hungarian_matching(gt_boxes_in, pred_boxes_in):
    # Compute the batch IoUs
    pred_boxes = pred_boxes_in.clone()
    gt_boxes = gt_boxes_in.clone()

    #print(pred_boxes, gt_boxes)

    gt_boxes[:, 0] = gt_boxes_in[:, 0]*360 / 2
    gt_boxes[:, 1] = gt_boxes_in[:, 1]*180 / 2
    gt_boxes[:, 2] = gt_boxes_in[:, 2]*90
    gt_boxes[:, 3] = gt_boxes_in[:, 3]*90

    gt_boxes = gt_boxes.to(torch.int)

    pred_boxes[:, 0] = pred_boxes_in[:, 0]*360 / 2
    pred_boxes[:, 1] = pred_boxes_in[:, 1]*180 / 2
    pred_boxes[:, 2] = pred_boxes_in[:, 2]*90
    pred_boxes[:, 3] = pred_boxes_in[:, 3]*90

    pred_boxes = pred_boxes.to(torch.int)

    #prediciton returning weird values
    #print(gt_boxes)
    iou_matrix = fov_iou_batch(gt_boxes, pred_boxes)

    print(iou_matrix)

    # Convert IoUs to cost
    cost_matrix = 1 - iou_matrix.detach().numpy()

    # Apply Hungarian matching
    gt_indices, pred_indices = linear_sum_assignment(cost_matrix)
    #print(gt_indices, pred_indices)

    # Extract the matched pairs
    matched_pairs = [(gt_boxes[i], pred_boxes[j]) for i, j in zip(gt_indices, pred_indices)]

    return matched_pairs, iou_matrix[gt_indices, pred_indices]

torch.cuda.empty_cache()
# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
num_epochs = 5000
learning_rate = 0.001
batch_size = 8
num_classes = 37


# Initialize dataset and dataloader
train_dataset = PascalVOCDataset(split='TRAIN', keep_difficult=False, max_images=80)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=train_dataset.collate_fn)

def init_weights(m):
    if isinstance(m, nn.Linear):
        # Choose the initialization method here (e.g., Xavier, He)
        init.xavier_uniform_(m.weight)
        if m.bias is not None:
            init.zeros_(m.bias)


model = SimpleObjectDetector(num_boxes=3, num_classes=num_classes).to(device)
#model.fc1.apply(init_weights)
#model.det_head.apply(init_weights)
#model.cls_head.apply(init_weights)
#model.conf_head.apply(init_weights)

optimizer = optim.Adam(model.parameters(), lr=learning_rate)

'''
conversão:



Para o primeiro conjunto 

[40,50,35,55]:
−2.443, −2.269, 0.611, 0.960

Para o segundo conjunto 
[ 35, 20, 37, 50]
−1.265,−1.396, 0.646, 0.873]'''



gt1 = torch.tensor(np.array([
    [-2.443+np.pi, -2.269+np.pi/2, 0.611, 0.960,0],
    [0.1250 * 2 * np.pi, 0.5560 * np.pi, 21.0000 / 180 * np.pi, 93.0000 / 180 * np.pi, 0],
    [0.4350 * 2 * np.pi, 0.5460 * np.pi, 13.0000 / 180 * np.pi, 69.0000 / 180 * np.pi, 0],
    [0.7300 * 2 * np.pi, 0.4810 * np.pi, 69.0000 / 180 * np.pi, 65.0000 / 180 * np.pi, 0],
    [0.4820 * 2 * np.pi, 0.4720 * np.pi,  5.0000 / 180 * np.pi, 29.0000 / 180 * np.pi, 0],
    [0.2960 * 2 * np.pi, 0.5590 * np.pi, 93.0000 / 180 * np.pi, 109.0000 / 180 * np.pi, 0]
]))


gt2 = torch.tensor(np.array([
            [-1.265+np.pi,-1.396+np.pi/2, 0.646, 0.873,0],
            [  0.4323 * 2 * np.pi,   0.5448* np.pi,  12.0000/180* np.pi,  68.0000/180* np.pi, 0],
            [  0.1224 *2 * np.pi,   0.5542* np.pi,  20.0000/180* np.pi,  92.0000/180* np.pi, 0],
            [  0.7286*2* np.pi,   0.4792* np.pi,  68.0000/180* np.pi,  64.0000/180* np.pi, 0],
            [  0.2948*2* np.pi,   0.5573* np.pi,  92.0000/180* np.pi, 108.0000/180* np.pi, 0],
            [  0.4802*2* np.pi,   0.4708* np.pi,   4.0000/180* np.pi,  28.0000/180* np.pi, 0]]))

for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    total_matches = 0 

    for i, (images, boxes_list, labels_list, confidences_list) in enumerate(train_loader):
        images = images.to(device)
        optimizer.zero_grad()
        # Forward pass
        detection_preds, classification_preds, confidence_preds = model(images)
        total_regression_loss = 0
        n=0

        for boxes, labels, det_preds, cls_preds, conf_preds in zip(boxes_list, labels_list, detection_preds, classification_preds, confidence_preds):
            boxes = boxes.to(device)
            det_preds = det_preds.to(device)          
            labels = labels.to(device)

            matches, matched_iou_scores = hungarian_matching(boxes[:5], boxes[5:10])
            regression_loss = (1 - matched_iou_scores).mean()

            total_regression_loss += regression_loss
            total_matches += len(matches)
            #if epoch>0 and epoch%2000==0:
            if True:
                #pass
                img1 = images[n].permute(1,2,0).cpu().numpy()*255
                img = process_and_save_image(img1, boxes.cpu().detach().numpy(), (0,255,0), f'/home/mstveras/images/img.png')
                img = process_and_save_image(img, det_preds.cpu().detach().numpy(), (255,0,0), f'/home/mstveras/images/img2.png')
            
                n+=1

        if total_matches > 0:
            avg_regression_loss = total_regression_loss / total_matches
        else:
            avg_regression_loss = torch.tensor(0, device=device)

        # Backward pass and optimize
        #avg_regression_loss.backward()
        optimizer.step()

        total_loss += avg_regression_loss.item()

    avg_epoch_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch}/{num_epochs}: Loss: {avg_epoch_loss}")

model_file = "best.pth"
torch.save(model.state_dict(), model_file)
print(f"Model saved to {model_file} after Epoch {epoch + 1}")
print('Training completed.')