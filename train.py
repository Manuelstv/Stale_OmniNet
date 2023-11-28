import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
#sfrom torchvision import transforms
import torch.nn.functional as F

# Import your model and dataset classes
from model import SimpleObjectDetectorWithBackbone
from datasets import PascalVOCDataset
torch.cuda.empty_cache()

import cv2
import numpy as np
import json
from numpy.linalg import norm
from skimage.io import imread
from calculate_RoIoU import Sph




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
    r = 100
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
    return Plotting.plotEquirectangular(image, np.vstack((u, v)).T, color)


def bbox_iou(pred_boxes, gt_boxes):
    """
    Calculate the IoU of two sets of boxes in (x_center, y_center, width, height) format.
    pred_boxes is a tensor of dimensions (N, 4)
    gt_boxes is a tensor of dimensions (M, 4)
    Returns a tensor of shape (N, M) where element (i, j) is the IoU of pred_boxes[i] and gt_boxes[j].
    """

    num_pred_boxes = pred_boxes.size(0)
    num_gt_boxes = gt_boxes.size(0)
    iou_matrix = torch.zeros((num_pred_boxes, num_gt_boxes), device=pred_boxes.device)

    for i in range(num_pred_boxes):
        for j in range(num_gt_boxes):
            # Convert from center format to corner format
            pred_box = pred_boxes[i]
            gt_box = gt_boxes[j]

            pred_box_x1 = pred_box[0] - pred_box[2] / 2
            pred_box_y1 = pred_box[1] - pred_box[3] / 2
            pred_box_x2 = pred_box[0] + pred_box[2] / 2
            pred_box_y2 = pred_box[1] + pred_box[3] / 2

            gt_box_x1 = gt_box[0] - gt_box[2] / 2
            gt_box_y1 = gt_box[1] - gt_box[3] / 2
            gt_box_x2 = gt_box[0] + gt_box[2] / 2
            gt_box_y2 = gt_box[1] + gt_box[3] / 2

            # Calculate intersection
            inter_x1 = torch.max(pred_box_x1, gt_box_x1)
            inter_y1 = torch.max(pred_box_y1, gt_box_y1)
            inter_x2 = torch.min(pred_box_x2, gt_box_x2)
            inter_y2 = torch.min(pred_box_y2, gt_box_y2)

            inter_area = max(inter_x2 - inter_x1 + 1, 0) * max(inter_y2 - inter_y1 + 1, 0)

            # Calculate union
            pred_box_area = (pred_box_x2 - pred_box_x1 + 1) * (pred_box_y2 - pred_box_y1 + 1)
            gt_box_area = (gt_box_x2 - gt_box_x1 + 1) * (gt_box_y2 - gt_box_y1 + 1)

            union_area = pred_box_area + gt_box_area - inter_area

            # Calculate IoU
            iou = inter_area / union_area
            iou_matrix[i, j] = iou

    return iou_matrix

def match_predictions_to_ground_truths(pred_boxes, gt_boxes, iou_threshold=0.5):
    iou_matrix = bbox_iou(pred_boxes, gt_boxes)  # Assuming bbox_iou can handle batched inputs

    matched_gt_indices = torch.argmax(iou_matrix, dim=1)
    max_ious = iou_matrix[range(len(pred_boxes)), matched_gt_indices]

    # Applying the IoU threshold
    matches = max_ious > iou_threshold
    matched_gt_indices[~matches] = -1  # Assign -1 for unmatched (or low IoU) predictions

    return matched_gt_indices



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

def iou_loss(pred_boxes, gt_boxes, matched_indices):
    """
    Compute the IoU loss for matched pairs of predicted and ground truth boxes.
    pred_boxes: Predicted boxes (N, 4)
    gt_boxes: Ground truth boxes (M, 4)
    matched_indices: Indices of matched ground truth boxes for each prediction
    """
    # Filter out the unmatched indices (-1 indicates unmatched)
    valid_indices = (matched_indices != -1)
    matched_pred_boxes = pred_boxes[valid_indices]
    matched_gt_boxes = gt_boxes[matched_indices[valid_indices]]

    iou = bbox_iou(matched_pred_boxes, matched_gt_boxes)
    iou_values = torch.diag(iou)  # Extract the IoUs for matched pairs
    loss = 1 - iou_values  # IoU loss is 1 - IoU
    return loss.mean()  # Return the mean IoU loss

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

x = 10  # Define the interval for printing the loss
background_class_label = 0

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

        n=0

        for boxes, labels, det_preds, cls_preds, conf_preds in zip(boxes_list, labels_list, detection_preds, classification_preds, confidence_preds):
            boxes = boxes.to(device)
            labels = labels.to(device)

            #print(det_preds)
            iou_matrix = bbox_iou(det_preds, boxes)

            color_map = {4: (0, 0, 255), 5: (0, 255, 0), 6: (255, 0, 0), 12: (255, 255, 0), 17: (0, 255, 255), 25: (255, 0, 255), 26: (128, 128, 0), 27: (0, 128, 128), 30: (128, 0, 128), 34: (128, 128, 128), 35: (64, 0, 0), 36: (0, 64, 0)}
            h, w = images.shape[:2]
            classes = labels

            img = images[n].permute(1, 2, 0).cpu().numpy()*255

            n+=1

            for i in range(len(boxes)):
                box = boxes[i].cpu()
                u00, v00, a_lat1, a_long1 = box[0], box[1], box[2], box[3]
                a_lat = np.radians(a_long1)
                a_long = np.radians(a_lat1)
                #color = color_map.get(classes[i], (255, 255, 255))
                img = plot_bfov(img, v00, u00, a_lat, a_long,(0,255,0), 300,600)
            #cv2.imwrite('/home/mstveras/final_image.png', img)

            # Determine best match for each prediction
            matched_indices = torch.argmax(iou_matrix, dim=1)
            max_ious = iou_matrix[range(len(det_preds)), matched_indices]
            
            # Apply IoU threshold to determine positive matches
            matched_gt_indices = matched_indices[max_ious > 0.1]
            unmatched_indices = matched_indices[max_ious <= 0.1]

            # Set confidence targets based on matching
            target_confidences = torch.zeros_like(conf_preds[:, 0])
            matched_gt_indices = matched_indices[matched_indices != -1]
            unmatched_gt_indices = matched_indices[matched_indices == -1]

            target_confidences[matched_gt_indices] = 1

            # Classification labels - background class for unmatched, actual labels for matched
            cls_labels = torch.full((len(cls_preds),), background_class_label, device=device, dtype=torch.long)

            cls_labels[matched_gt_indices] = labels[matched_indices[matched_indices != -1]]

            # Compute losses
            regression_loss = iou_loss(det_preds, boxes, matched_indices)

            classification_loss = classification_criterion(cls_preds, cls_labels)
            confidence_loss = confidence_criterion(conf_preds[:, 0], target_confidences)

            # Accumulate losses
            total_regression_loss += regression_loss
            total_confidence_loss += confidence_loss
            total_classification_loss += classification_loss

        # Compute the average losses
        avg_regression_loss = total_regression_loss / len(boxes_list)
        avg_classification_loss = total_classification_loss / len(boxes_list)
        avg_confidence_loss = total_confidence_loss / len(boxes_list)

        total_loss = avg_regression_loss + avg_classification_loss + avg_confidence_loss
        print(f"Epoch {epoch}/{num_epochs}: Loss: {total_loss}")

        # Backward and optimize
        total_loss.backward()
        optimizer.step()

         

model_file = f"best.pth"
torch.save(model.state_dict(), model_file)
print(f"Model saved to {model_file} after Epoch {epoch + 1}")

print('Training completed.')
