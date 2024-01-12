import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
import torch.nn.init as init

from model import SimpleObjectDetector, SimpleObjectDetectorMobile, SimpleObjectDetectorResnet
from datasets import PascalVOCDataset

import cv2
import json
from sphiou import Sph
from plot_tools import process_and_save_image, process_and_save_image_planar
from foviou import fov_iou, deg2rad, angle2radian, fov_giou_loss, iou
from sph2pob import sph_iou_aligned, fov_iou_aligned

def fov_iou_batch(gt_boxes, pred_boxes):
    """
    Calculate the Intersection over Union (IoU) for each pair of ground truth and predicted boxes.

    Args:
    - gt_boxes (Tensor): A tensor of ground truth bounding boxes.
    - pred_boxes (Tensor): A tensor of predicted bounding boxes.

    Returns:
    - Tensor: A matrix of IoU values, where each element [i, j] is the IoU of the ith ground truth box and the jth predicted box.
    """
    # Initialize a tensor to store IoU values
    ious = torch.zeros((len(gt_boxes), len(pred_boxes)))

    # Iterate over each ground truth and predicted box pair
    for i, Bg in enumerate(gt_boxes):
        for j, Bd in enumerate(pred_boxes):
            #ious[i, j] = fov_iou(deg2rad(Bg), deg2rad(Bd))
            ious[i,j] = iou(Bg,Bd)
    return ious

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
    pred_boxes = pred_boxes_in.clone()
    gt_boxes = gt_boxes_in.clone()

    gt_boxes[:, 0] = gt_boxes_in[:, 0]
    gt_boxes[:, 1] = gt_boxes_in[:, 1]
    gt_boxes[:, 2] = gt_boxes_in[:, 2]
    gt_boxes[:, 3] = gt_boxes_in[:, 3]

    gt_boxes = gt_boxes.to(torch.float)

    pred_boxes[:, 0] = pred_boxes_in[:, 0]
    pred_boxes[:, 1] = pred_boxes_in[:, 1]
    pred_boxes[:, 2] = pred_boxes_in[:, 2]
    pred_boxes[:, 3] = pred_boxes_in[:, 3]

    pred_boxes = pred_boxes.to(torch.float)
    iou_matrix = fov_iou_batch(gt_boxes, pred_boxes)

    # Convert IoUs to cost
    cost_matrix = 1 - iou_matrix.detach().numpy()

    # Apply Hungarian matching
    gt_indices, pred_indices = linear_sum_assignment(cost_matrix)

    # Extract the matched pairs
    matched_pairs = [(gt_boxes[i], pred_boxes[j]) for i, j in zip(gt_indices, pred_indices)]

    return matched_pairs, iou_matrix[gt_indices, pred_indices]

def init_weights(m):
    """
    Initialize the weights of a linear layer using Xavier uniform initialization.

    Args:
    - m (nn.Module): A linear layer of a neural network.

    Note:
    - This function is designed to be applied to a linear layer of a PyTorch model.
    - If the layer has a bias term, it will be initialized to zero.
    """
    if isinstance(m, nn.Linear):
        # Choose the initialization method here (e.g., Xavier, He)
        init.xavier_uniform_(m.weight)
        if m.bias is not None:
            init.zeros_(m.bias)

def train_one_epoch(epoch, train_loader, model, optimizer, device, new_w, new_h):
    model.train()
    total_loss = 0
    total_matches = 0

    for i, (images, boxes_list, labels_list, confidences_list) in enumerate(train_loader):
        images, losses, n = images.to(device), [], 0
        optimizer.zero_grad()
        detection_preds = model(images)

        for boxes, labels, det_preds in process_batches(boxes_list, labels_list, detection_preds, device, new_w, new_h, epoch, n, images):
            matches, matched_iou_scores = hungarian_matching(boxes, det_preds)
            regression_loss = (1 - matched_iou_scores).mean()
            losses.append(regression_loss)
            total_matches += len(matches)

        update_model(losses, total_matches, total_loss, optimizer)

    return total_loss / len(train_loader)

def process_batches(boxes_list, labels_list, detection_preds, device, new_w, new_h, epoch, n, images):
    for boxes, labels, det_preds in zip(boxes_list, labels_list, detection_preds):
        boxes, det_preds, labels = boxes.to(device), det_preds.to(device), labels.to(device)
        save_images(epoch, boxes, det_preds, new_w, new_h, n, images)
        n += 1
        yield boxes, labels, det_preds

def save_images(epoch, boxes, det_preds, new_w, new_h, n, images):
    if epoch > 0 and epoch % 100 == 0:
        img1 = images[n].mul(255).clamp(0, 255).permute(1, 2, 0).cpu().numpy().astype(np.uint8).copy()
        draw_boxes(img1, boxes, (0, 255, 0), new_w, new_h)
        draw_boxes(img1, det_preds, (255, 0, 0), new_w, new_h)
        cv2.imwrite(f'/home/mstveras/images/img{n}.jpg', img1)

def draw_boxes(image, boxes, color, new_w, new_h):
    for box in boxes:
        x_min, y_min, x_max, y_max = [int(box[i] * new_w if i % 2 == 0 else box[i] * new_h) for i in range(4)]
        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), color)

def update_model(losses, total_matches, total_loss, optimizer):
    if total_matches > 0:
        avg_regression_loss = sum(losses) / total_matches
        avg_regression_loss.backward()
        optimizer.step()
        total_loss += avg_regression_loss.item()
    else:
        print('No matches found, not backpropagating.')

def validate_model(epoch, val_loader, model, device, best_val_loss):
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for images, boxes_list, labels_list, confidences_list in val_loader:
            images = images.to(device)
            detection_preds = model(images)

            val_losses = [process_validation(boxes, det_preds, labels, device) 
                          for boxes, labels, det_preds in zip(boxes_list, labels_list, detection_preds)]

            val_loss += sum(val_losses) / len(val_losses) if val_losses else 0

    avg_val_loss = val_loss / len(val_loader)
    print(f"Epoch {epoch}/5: Validation Loss: {avg_val_loss}")

    if avg_val_loss < best_val_loss:
        save_best_model(epoch, avg_val_loss, model)

def process_validation(boxes, det_preds, labels, device):
    boxes, det_preds, labels = boxes.to(device), det_preds.to(device), labels.to(device)
    matches, matched_iou_scores = hungarian_matching(boxes, det_preds)
    regression_loss = (1 - matched_iou_scores).mean()
    return regression_loss

def save_best_model(epoch, avg_val_loss, model):
    best_val_loss = avg_val_loss
    best_model_file = f"best_epoch_{epoch}.pth"
    torch.save(model.state_dict(), best_model_file)
    print(f"Model saved to {best_model_file} with Validation Loss: {avg_val_loss}")

if __name__ == "__main__":

    torch.cuda.empty_cache()

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Hyperparameters
    num_epochs = 5
    learning_rate = 0.001
    batch_size = 10
    num_classes = 3
    max_images = 10
    num_boxes = 3
    best_val_loss = float('inf')

    new_w, new_h = 600,300

    # Initialize dataset and dataloader
    train_dataset = PascalVOCDataset(split='TRAIN', keep_difficult=False, max_images=max_images)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=train_dataset.collate_fn)

    val_dataset = PascalVOCDataset(split='VAL', keep_difficult=False, max_images=max_images)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, collate_fn=train_dataset.collate_fn)

    model = SimpleObjectDetector(num_boxes=num_boxes, num_classes=num_classes).to(device)
    model.fc1.apply(init_weights)
    model.det_head.apply(init_weights)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(5):
        avg_epoch_loss = train_one_epoch(epoch, train_loader, model, optimizer, device, new_w, new_h)
        print(f"Epoch {epoch}/5: Loss: {avg_epoch_loss}")

        validate_model(epoch, val_loader, model, device, best_val_loss)

    print('Training and validation completed.')