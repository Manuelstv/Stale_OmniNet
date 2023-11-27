import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
#sfrom torchvision import transforms
import torch.nn.functional as F

# Import your model and dataset classes
from model import SimpleObjectDetectorWithBackbone
from datasets import PascalVOCDataset
torch.cuda.empty_cache()

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
num_epochs = 10
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

        for boxes, labels, det_preds, cls_preds, conf_preds in zip(boxes_list, labels_list, detection_preds, classification_preds, confidence_preds):
            boxes = boxes.to(device)
            labels = labels.to(device)

            #print(det_preds)
            # Calculate IoU matrix
            iou_matrix = bbox_iou(det_preds, boxes)
            #print(iou_matrix)
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
            #print(target_confidences)

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
