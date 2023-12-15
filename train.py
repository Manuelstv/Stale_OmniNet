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

from model import SimpleObjectDetectorWithBackbone, SimpleObjectDetector, SimpleObjectDetectorWithBackbone2
from datasets import PascalVOCDataset
torch.cuda.empty_cache()

import cv2
import json
from calculate_RoIoU import Sph
from plot_tools import process_and_save_image
from foviou import fov_iou, deg_to_rad
import math

def fov_iou_batch(gt_boxes, pred_boxes):
    # Initialize a tensor to store IoU values
    ious = torch.zeros((len(gt_boxes), len(pred_boxes)))

    # Iterate over each ground truth and predicted box pair
    for i, Bg in enumerate(gt_boxes):
        for j, Bd in enumerate(pred_boxes):
            ious[i, j] = fov_iou(deg_to_rad(Bg), deg_to_rad(Bd))
            if ious[i,j]>1:
                print(ious[i,j])
                #print(Bg, Bd)
                #break
                pass

    return ious

def deg_to_rad(degrees):
    return [math.radians(degree) for degree in degrees]

def hungarian_matching(gt_boxes_in, pred_boxes_in):
    # Compute the batch IoUs
    pred_boxes = pred_boxes_in.clone()
    gt_boxes = gt_boxes_in.clone()

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

    '''
    coordinates = [
    ([40, 50, 35, 55,0], [35, 20, 37, 50,0]),
    ([30, 60, 60, 60,0], [55, 40, 60, 60,0]),
    ([50, -78, 25, 46,0], [30, -75, 26, 45,0]),
    ([30, 75, 30, 60,0], [60, 40, 60, 60,0]),
    ([40, 70, 25, 30,0], [60, 85, 30, 30,0]),
    ([30, 75, 30, 30,0], [60, 55, 40, 50,0])
]

    # Convert the coordinates to tensors
    gt_boxes = [torch.tensor(b1, dtype=torch.int) for b1, _ in coordinates]
    pred_boxes = [torch.tensor(b2, dtype=torch.int) for _, b2 in coordinates]'''


    #print(gt_boxes, pred_boxes)
    #prediciton returning weird values
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
batch_size = 8
num_classes = 37
max_images = 10


# Initialize dataset and dataloader
train_dataset = PascalVOCDataset(split='TRAIN', keep_difficult=False, max_images=max_images)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=train_dataset.collate_fn)

def init_weights(m):
    if isinstance(m, nn.Linear):
        # Choose the initialization method here (e.g., Xavier, He)
        init.xavier_uniform_(m.weight)
        if m.bias is not None:
            init.zeros_(m.bias)


model = SimpleObjectDetector(num_boxes=30, num_classes=num_classes).to(device)
model.fc1.apply(init_weights)
model.det_head.apply(init_weights)
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

for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    total_matches = 0
    total_regression_loss = torch.tensor(0.0, device=device, requires_grad=True)

    for i, (images, boxes_list, labels_list, confidences_list) in enumerate(train_loader):
        images = images.to(device)
        optimizer.zero_grad()
        detection_preds, classification_preds, confidence_preds = model(images)
        n=0

        for boxes, labels, det_preds, cls_preds, conf_preds in zip(boxes_list, labels_list, detection_preds, classification_preds, confidence_preds):


            boxes = torch.tensor([[-0.0246,  0.0070,  0.4983,  0.5012,  0.0000],
                        [ 0.0059,  0.0295,  0.4942,  0.5054,  0.0000],
                        [ 0.0233,  0.0342,  0.4982,  0.5016,  0.0000],
                        [-0.0490,  0.0103,  0.5090,  0.4972,  0.0000],
                        [ 0.0585,  0.0395,  0.4941,  0.5064,  0.0000],
                        [-0.0402, -0.0128,  0.5089,  0.5137,  0.0000],
                        [ 0.0286,  0.0039,  0.4867,  0.4976,  0.0000],
                        [ 0.0484, -0.0290,  0.5117,  0.5019,  0.0000],
                        [ 0.0265,  0.0019,  0.5127,  0.4982,  0.0000],
                        [ 0.0426, -0.0630,  0.4873,  0.5001,  0.0000],
                        [ 0.0398, -0.0331,  0.4955,  0.4983,  0.0000],
                        [-0.0189,  0.0334,  0.4973,  0.4879,  0.0000],
                        [-0.0469,  0.0487,  0.4865,  0.5119,  0.0000],
                        [ 0.0462, -0.0444,  0.4981,  0.5106,  0.0000],
                        [-0.0007,  0.0522,  0.4935,  0.5072,  0.0000],
                        [ 0.0268, -0.0203,  0.5037,  0.5020,  0.0000],
                        [-0.0356,  0.0612,  0.4987,  0.5000,  0.0000],
                        [ 0.0457,  0.0208,  0.4936,  0.4997,  0.0000],
                        [ 0.0657, -0.0451,  0.4941,  0.5072,  0.0000],
                        [ 0.0484, -0.0519,  0.5105,  0.5066,  0.0000],
                        [ 0.0457,  0.0456,  0.4852,  0.4825,  0.0000],
                        [ 0.0108, -0.0567,  0.5030,  0.5091,  0.0000],
                        [-0.0509,  0.0018,  0.5011,  0.4958,  0.0000]])
            
            boxes = boxes.to(device)
            det_preds = det_preds.to(device)          
            labels = labels.to(device)
            matches, matched_iou_scores = hungarian_matching(boxes, det_preds)

            #sum or mean??????????
            #print(matched_iou_scores)
            regression_loss = (1 - matched_iou_scores).mean()

            total_regression_loss = total_regression_loss + regression_loss
            total_matches += len(matches)
            
            if epoch>0 and epoch%10==0:
            #if True:
                pass
                #img1 = images[n].permute(1,2,0).cpu().numpy()*255
                #img = process_and_save_image(img1, boxes.cpu().detach().numpy(), (0,255,0), f'/home/mstveras/images/img.png')
                #img = process_and_save_image(img, det_preds.cpu().detach().numpy(), (255,0,0), f'/home/mstveras/images/img2.png')
            
                #n+=1

        if total_matches > 0:
            avg_regression_loss = total_regression_loss / total_matches
        else:
            avg_regression_loss = total_regression_loss*0

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