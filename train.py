import json
import os

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torch.optim as optim
from scipy.optimize import linear_sum_assignment
from torch.utils.data import DataLoader

from datasets import PascalVOCDataset
from foviou import (deg2rad, fov_iou, fov_giou_loss, iou, angle2radian)
from model import (SimpleObjectDetector, SimpleObjectDetectorMobile,
                   SimpleObjectDetectorResnet)
from plot_tools import process_and_save_image, process_and_save_image_planar
from sphiou import Sph
from sph2pob import sph_iou_aligned


def fov_iou_batch(gt_boxes, pred_boxes):
    """
    Calculate the Intersection over Union (IoU) for each pair of ground truth and predicted boxes.

    Args:
    - gt_boxes (Tensor): A tensor of ground truth bounding boxes.
    - pred_boxes (Tensor): A tensor of predicted bounding boxes.

    Returns:
    - Tensor: A matrix of IoU values, where each element [i, j] is the IoU of the ith ground truth box and the jth predicted box.
    """
    # Initialize a tensor to store IoU values, on the same device as the input boxes
    iou_values = []
    for Bg in gt_boxes:
        row = []
        for Bd in pred_boxes:
            row.append(iou(Bg, Bd))
        iou_values.append(row)

    # Convert list of lists to a tensor
    ious = torch.tensor(iou_values, device=gt_boxes.device, dtype=torch.float32)
    return ious.requires_grad_()

def hungarian_matching(gt_boxes_in, pred_boxes_in):
    """
    Perform Hungarian matching between ground truth and predicted boxes to find the best match based on IoU scores.

    Args:
    - gt_boxes_in (Tensor): A tensor of ground truth bounding boxes.
    - pred_boxes_in (Tensor): A tensor of predicted bounding boxes.

    Returns:
    - list of tuples: Matched pairs of ground truth and predicted boxes.
    - Tensor: IoU scores for the matched pairs.
    """
    # Compute the batch IoUs
    gt_boxes = gt_boxes_in.clone().to(torch.float)
    pred_boxes = pred_boxes_in.clone().to(torch.float)

    iou_matrix = fov_iou_batch(gt_boxes, pred_boxes)

    # Convert IoUs to cost
    cost_matrix = 1 - iou_matrix.detach().cpu().numpy()

    # Apply Hungarian matching
    gt_indices, pred_indices = linear_sum_assignment(cost_matrix)

    # Extract the matched pairs
    matched_pairs = [(gt_indices[i], pred_indices[i]) for i in range(len(gt_indices))]


    return matched_pairs, iou_matrix

def init_weights(m):
    """
    Initialize the weights of a linear layer.

    Args:
    - m (nn.Module): A linear layer of a neural network.

    Note:
    - This function is designed to be applied to a linear layer of a PyTorch model.
    - If the layer has a bias term, it will be initialized to zero.
    """
    if isinstance(m, nn.Linear):
        nn.init.uniform_(m.weight, -10, 10)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)

def train_one_epoch(epoch, train_loader, model, optimizer, device, new_w, new_h):
    model.train()
    total_loss = torch.tensor(0.0, device=device, requires_grad=True)
    avg_epoch_loss = 0
    total_matches =0

    for i, (images, boxes_list, labels_list, confidences_list) in enumerate(train_loader):
        images = images.to(device)
        optimizer.zero_grad()
        detection_preds = model(images)

        batch_loss = 0
        total_matches = 0
        for boxes, labels, det_preds in process_batches(boxes_list, labels_list, detection_preds, device, new_w, new_h, epoch, i, images):
            matches, iou_scores = hungarian_matching(boxes, det_preds)
            if len(matches) > 0:
                matched_iou_scores = iou_scores[[match[0] for match in matches], [match[1] for match in matches]]
                batch_loss += (1 - matched_iou_scores).mean()
                total_matches += len(matches)

        if total_matches > 0:
            batch_loss /= total_matches

        batch_loss.backward()
        optimizer.step()

        total_loss = total_loss + batch_loss.item()

    print(f"Epoch {epoch}: Train Loss: {total_loss}")


def validate_model(epoch, val_loader, model, device, best_val_loss):
    model.eval()
    total_val_loss = 0
    total_matches=0

    with torch.no_grad():
        for i, (images, boxes_list, labels_list, confidences_list) in enumerate(val_loader):
            images = images.to(device)
            detection_preds = model(images)

            for boxes, labels, det_preds in process_batches(boxes_list, labels_list, detection_preds, device, new_w, new_h, epoch, i, images):
                matches, iou_scores = hungarian_matching(boxes, det_preds)
                
                gt_indices = [match[0] for match in matches]
                pred_indices = [match[1] for match in matches]

                gt_indices = torch.tensor(gt_indices, dtype=torch.long, device=iou_scores.device)
                pred_indices = torch.tensor(pred_indices, dtype=torch.long, device=iou_scores.device)
                
                matched_iou_scores = iou_scores[gt_indices, pred_indices]
                iou_loss = (1 - matched_iou_scores).mean()
                #n=i
                #save_images(boxes, det_preds, new_w, new_h, n, images)
                
            total_val_loss += iou_loss.item()

    avg_val_loss = total_val_loss / len(val_loader)
    print(f"Epoch {epoch}: Validation Loss: {avg_val_loss}")

    # Update best validation loss and save model if necessary
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        save_best_model(epoch, model)

    return best_val_loss, epoch

def process_batches(boxes_list, labels_list, detection_preds, device, new_w, new_h, epoch, n, images):
    for boxes, labels, det_preds in zip(boxes_list, labels_list, detection_preds):
        boxes, det_preds, labels = boxes.to(device), det_preds.to(device), labels.to(device)
        #save_images(boxes, det_preds, new_w, new_h, n, images)
        n += 1
        yield boxes, labels, det_preds

def save_images(boxes, det_preds, new_w, new_h, n, images):
    img1 = images[n].mul(255).clamp(0, 255).permute(1, 2, 0).cpu().numpy().astype(np.uint8).copy()
    draw_boxes(img1, boxes, (0, 255, 0), new_w, new_h)
    draw_boxes(img1, det_preds, (255, 0, 0), new_w, new_h)
    cv2.imwrite(f'/home/mstveras/images/img{n}.jpg', img1)

def draw_boxes(image, boxes, color, new_w, new_h):
    for box in boxes:
        x_min, y_min, x_max, y_max = [int(box[i] * new_w if i % 2 == 0 else box[i] * new_h) for i in range(4)]
        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), color)

def save_best_model(epoch, model):
    #best_val_loss = avg_val_loss
    best_model_file = f"best.pth"
    torch.save(model.state_dict(), best_model_file)
    #print(f"Model saved to {best_model_file} with Validation Loss: {avg_val_loss}")

if __name__ == "__main__":

    torch.cuda.empty_cache()

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Hyperparameters
    num_epochs = 500
    learning_rate = 0.001
    batch_size = 8
    num_classes = 1
    max_images = 8
    num_boxes = 10
    best_val_loss = float('inf')
    new_w, new_h = 600,300

    # Initialize dataset and dataloader
    train_dataset = PascalVOCDataset(split='TRAIN', keep_difficult=False, max_images=max_images, new_w = new_w, new_h = new_h)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=train_dataset.collate_fn)

    #val_dataset = PascalVOCDataset(split='VAL', keep_difficult=False, max_images=max_images, new_w = new_w, new_h = new_h)
    #val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, collate_fn=train_dataset.collate_fn)

    model = SimpleObjectDetector(num_boxes=num_boxes, num_classes=num_classes).to(device)
    model.det_head.apply(init_weights)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        train_one_epoch(epoch, train_loader, model, optimizer, device, new_w, new_h)
        #validate_model(epoch, val_loader, model, device, best_val_loss)

    print('Training and validation completed.')