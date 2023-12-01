import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
#sfrom torchvision import transforms
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

# Import your model and dataset classes
from model import SimpleObjectDetectorWithBackbone
from datasets import PascalVOCDataset
torch.cuda.empty_cache()

import cv2
import json
from numpy.linalg import norm
from skimage.io import imread
from calculate_RoIoU_torch import Sph


class Rotation:
    @staticmethod
    def Rx(alpha):
        return np.asarray([[1, 0, 0], [0, np.cos(alpha), -np.sin(alpha)], [0, np.sin(alpha), np.cos(alpha)]])
    @staticmethod
    def Ry(beta):
        return np.asarray([[np.cos(beta), 0, np.sin(beta)], [0, 1, 0], [-np.sin(beta), 0, np.cos(beta)]])
    @staticmethod
    def Rz(gamma):
        return np.asarray([[np.cos(gamma), -np.sin(gamma), 0], [np.sin(gamma), np.cos(gamma), 0], [0, 0, 1]])

class Plotting:
    @staticmethod
    def plotEquirectangular(image, kernel, color):
        resized_image = np.ascontiguousarray(image, dtype=np.uint8)
        kernel = kernel.astype(np.int32)
        hull = cv2.convexHull(kernel)
        cv2.polylines(resized_image, [hull], isClosed=True, color=color, thickness=2)
        return resized_image

def plot_bfov(image, v00, u00, a_lat, a_long, color, h, w):
    phi00 = (u00 - w / 2.) * ((2. * np.pi) / w)
    theta00 = -(v00 - h / 2.) * (np.pi / h)
    r = 300
    d_lat = r / (2 * np.tan(a_lat / 2))
    d_long = r / (2 * np.tan(a_long / 2))
    p = []
    for i in range(-(r - 1) // 2, (r + 1) // 2):
        for j in range(-(r - 1) // 2, (r + 1) // 2):
            p += [np.asarray([i * d_lat / d_long, j, d_lat])]
    R = np.dot(Rotation.Ry(phi00), Rotation.Rx(theta00))
    p = np.asarray([np.dot(R, (p[ij] / norm(p[ij]))) for ij in range(r * r)])
    phi = np.asarray([np.arctan2(p[ij][0], p[ij][2]) for ij in range(r * r)])
    theta = np.asarray([np.arcsin(p[ij][1]) for ij in range(r * r)])
    u = (phi / (2 * np.pi) + 1. / 2.) * w
    v = h - (-theta / np.pi + 1. / 2.) * h

    # New logic for wrapping around
    u_wrapped = np.mod(u, w)  # Wrap around for u coordinates

    # Identify points that need to be split
    split_indices = np.where(np.abs(np.diff(u_wrapped)) >= w / 2)[0]

    # Plot the parts that do not require splitting
    if len(split_indices) == 0:
        return Plotting.plotEquirectangular(image, np.vstack((u_wrapped, v)).T, color)

    # Plotting the parts that require splitting
    for idx in range(len(split_indices) + 1):
        start_idx = split_indices[idx - 1] + 1 if idx > 0 else 0
        end_idx = split_indices[idx] + 1 if idx < len(split_indices) else len(u_wrapped)
        Plotting.plotEquirectangular(image, np.vstack((u_wrapped[start_idx:end_idx], v[start_idx:end_idx])).T, color)

    return image

def process_and_save_image(images, boxes, image_index, color, save_path):
    """
    Process an image from a list, plot bounding boxes, and save the image.
    
    Args:
    - images: A list of images.
    - boxes: A list of bounding boxes.
    - image_index: The index of the image in the images list to process.
    - file_path: The path to save the processed image.
    """
    # Process the image

    # Plot each bounding box
    for box in boxes:
        box = box
        u00, v00, a_lat1, a_long1 = 600*box[0], 300*box[1], 180*box[2], 180*box[3]
        a_lat = np.radians(a_long1)
        a_long = np.radians(a_lat1)
        images = plot_bfov(images, v00, u00, a_lat, a_long, color, 300, 600)

    # Save the processed image
    cv2.imwrite(save_path, images)
    return images

def transFormat(gt):
    '''
    Change the format and range of the RBFoV Representations.
    Input:
    - gt: the last dimension: [center_x, center_y, fov_x, fov_y, angle]
          center_x : [0,1]
          center_y : [0,1]
          fov_x    : [0, 180]
          fov_y    : [0, 180]
          All parameters are angles.
    Output:
    - ann: the last dimension: [center_x', center_y', fov_x', fov_y', angle]
           center_x' : [0, 2 * pi]
           center_y' : [0, pi]
           fov_x'    : [0, pi]
           fov_y'    : [0, pi]
           All parameters are radians.
    '''
    import copy
    ann = copy.copy(gt)
    ann[..., 2] = ann[..., 2] * np.pi
    ann[..., 3] = ann[..., 3] * np.pi
    ann[..., 0] = ann[..., 0] *2*np.pi
    ann[..., 1] = ann[...,1]*np.pi

    return ann

class SphericalIoULoss(nn.Module):
    def __init__(self):
        super(SphericalIoULoss, self).__init__()

    def forward(self, preds, targets):
        """
        Compute the spherical IoU loss.

        Args:
        - preds: Predicted bounding boxes, [N, 5] (center_x, center_y, fov_x, fov_y, angle)
        - targets: Ground truth bounding boxes, [N, 5] (center_x, center_y, fov_x, fov_y, angle)

        Returns:
        - loss: Mean spherical IoU loss
        """
        # Ensure preds and targets are tensors
        preds = torch.tensor(preds, dtype=torch.float32)
        targets = torch.tensor(targets, dtype=torch.float32)

        # Calculate spherical IoU
        sph_iou = Sph().sphIoU(preds, targets)

        # IoU loss is 1 - IoU
        loss = 1 - sph_iou

        return loss.mean()

def match_bfov_with_hungarian(spherical_iou_matrix):
    # Convert the IoU matrix to a cost matrix for the Hungarian algorithm
    cost_matrix = 1 - spherical_iou_matrix

    # Apply the Hungarian algorithm (linear_sum_assignment) to the cost matrix
    row_indices, col_indices = linear_sum_assignment(cost_matrix)

    # Each pair of (row_indices[i], col_indices[i]) represents a match
    matches = list(zip(row_indices, col_indices))

    # Returns a list of tuples; each tuple contains the indices of the matched predictions and ground truths
    return matches

torch.cuda.empty_cache()
# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
num_epochs = 5
learning_rate = 0.0001
batch_size = 8
num_classes = 37


# Initialize dataset and dataloader
train_dataset = PascalVOCDataset(split='TRAIN', keep_difficult=False, max_images=400)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=train_dataset.collate_fn)

#val_dataset = PascalVOCDataset(split='VAL', keep_difficult=False, max_images=200)  # Adjust max_images as needed
#val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=val_dataset.collate_fn)

model = SimpleObjectDetectorWithBackbone(num_boxes=30, num_classes=num_classes).to(device)

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, num_classes=37):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.num_classes = num_classes

    def forward(self, inputs, targets):
        # Convert targets to one-hot encoding
        targets_one_hot = F.one_hot(targets, num_classes=self.num_classes).float()
        
        # Compute the binary cross-entropy loss
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets_one_hot, reduction='none')

        # Apply the focal loss formula
        pt = torch.exp(-BCE_loss)  # prevents nans when probability 0
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss

        return F_loss.mean()

classification_criterion = FocalLoss()
confidence_criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)


gt2 = np.array([
            [  0.1224 *2 * np.pi,   0.5542* np.pi,  20.0000/180* np.pi,  92.0000/180* np.pi, 0],
            [  0.4323 * 2 * np.pi,   0.5448* np.pi,  12.0000/180* np.pi,  68.0000/180* np.pi, 0],
            [  0.7286*2* np.pi,   0.4792* np.pi,  68.0000/180* np.pi,  64.0000/180* np.pi, 0],
            [  0.4802*2* np.pi,   0.4708* np.pi,   4.0000/180* np.pi,  28.0000/180* np.pi, 0],
            [  0.2948*2* np.pi,   0.5573* np.pi,  92.0000/180* np.pi, 108.0000/180* np.pi, 0]])

x = 10  # Define the interval for printing the loss
background_class_label = 0

sph_iou_loss_fn = SphericalIoULoss()

for epoch in range(num_epochs):
    model.train()

    for i, (images, boxes_list, labels_list, confidences_list) in enumerate(train_loader):
        images = images.to(device)

        optimizer.zero_grad()

        # Forward pass with the whole batch
        detection_preds, classification_preds, confidence_preds = model(images)

        # Initialize total losses
        total_regression_loss = 0
        total_confidence_loss = 0
        total_classification_loss = 0

        for boxes, labels, det_preds, cls_preds, conf_preds in zip(boxes_list, labels_list, detection_preds, classification_preds, confidence_preds):
            boxes = boxes.to(device)

            gt = det_preds.to(device)

            labels = labels.to(device)

            '''
            img1 = images[0].permute(1,2,0).cpu().numpy()*255

            #@images = images.cpu().numpy() * 255
            img = process_and_save_image(img1, gt, 0, (0,255,0), f'/home/mstveras/img.png')

            img = process_and_save_image(img, boxes, 0, (255,0,0), f'/home/mstveras/img2.png')

            #zeros_column = np.zeros((gt.shape[0], 1))

            # Stack the zeros column with the original array
            #gt = np.hstack((gt, zeros_column))
            #zeros_column = np.zeros((boxes.shape[0], 1))

            # Stack the zeros column with the original array
            #boxes = np.hstack((boxes, zeros_column))'''

            sphIoU = Sph().sphIoU(boxes, gt)
            print(sphIoU)

            #each tuple contains the indices of the matched predictions and ground truths
            matches = match_bfov_with_hungarian(sphIoU)

            total_regression_loss = 0
            total_classification_loss = 0
            total_confidence_loss = 0
            '''
            for pred_idx, gt_idx in matches:
                matched_pred = det_preds[pred_idx].unsqueeze(0)  # Add batch dimension
                matched_gt = boxes[gt_idx].unsqueeze(0)         # Add batch dimension

                # Calculate the spherical IoU loss
                regression_loss = sph_iou_loss_fn(matched_pred, matched_gt)

                # Accumulate the regression loss
                total_regression_loss += regression_loss.item()

            matched_gt_indices = []
            matched_pred_indices = []

            for pred_idx, gt_idx in matches:
                matched_pred_indices.append(pred_idx)
                matched_gt_indices.append(gt_idx)

            matched_pred_indices = torch.tensor(matched_pred_indices, device=device)
            matched_gt_indices = torch.tensor(matched_gt_indices, device=device)

            # Extract the matched predictions and ground truths
            matched_preds = det_preds[matched_pred_indices]
            matched_gts = boxes[matched_gt_indices]

            # Classification and confidence targets
            cls_labels = labels[matched_gt_indices]
            target_confidences = torch.ones_like(cls_labels, dtype=torch.float, device=device)  # Assuming 1 for object presence

            # Compute classification and confidence losses
            classification_loss = classification_criterion(cls_preds[matched_pred_indices], cls_labels)
            confidence_loss = confidence_criterion(conf_preds[matched_pred_indices, 0], target_confidences)

            # Accumulate losses
            total_regression_loss += regression_loss
            total_confidence_loss += confidence_loss
            total_classification_loss += classification_loss'''

        # Compute the average losses
        avg_regression_loss = total_regression_loss / len(boxes_list)
        avg_classification_loss = total_classification_loss / len(boxes_list)
        avg_confidence_loss = total_confidence_loss / len(boxes_list)

        total_loss = avg_regression_loss + avg_classification_loss + avg_confidence_loss
        print(f"Epoch {epoch}/{num_epochs}: Loss: {total_loss}")

        # Backward and optimize
        t#otal_loss.backward()
        optimizer.step()

         

model_file = f"best.pth"
torch.save(model.state_dict(), model_file)
print(f"Model saved to {model_file} after Epoch {epoch + 1}")

print('Training completed.')
