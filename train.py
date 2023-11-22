import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
#sfrom torchvision import transforms

# Import your model and dataset classes
from model import SimpleObjectDetector
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

def iou_loss(preds, targets):
    """
    Calculate the IoU loss for the predicted and target bounding boxes.
    """
    iou = bbox_iou(preds, targets)
    loss = 1 - iou  # IoU loss is 1 - IoU
    return loss.mean()


def match_predictions_to_ground_truths(pred_boxes, gt_boxes, iou_threshold=0.5):
    iou_matrix = bbox_iou(pred_boxes, gt_boxes)  # Assuming bbox_iou can handle batched inputs

    matched_gt_indices = torch.argmax(iou_matrix, dim=1)
    max_ious = iou_matrix[range(len(pred_boxes)), matched_gt_indices]

    # Applying the IoU threshold
    matches = max_ious > iou_threshold
    matched_gt_indices[~matches] = -1  # Assign -1 for unmatched (or low IoU) predictions

    return matched_gt_indices


torch.cuda.empty_cache()
# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
num_epochs = 35
learning_rate = 0.0001
batch_size = 16
num_classes = 38


# Initialize dataset and dataloader
train_dataset = PascalVOCDataset(split='TRAIN', keep_difficult=False, max_images=100)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=train_dataset.collate_fn)

val_dataset = PascalVOCDataset(split='VAL', keep_difficult=False, max_images=200)  # Adjust max_images as needed
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=val_dataset.collate_fn)

model = SimpleObjectDetector(num_boxes=50, num_classes=num_classes).to(device)

# Loss and optimizer
classification_criterion = nn.CrossEntropyLoss()  # For class predictions
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
            #labels = labels - 1  # Adjust labels if necessary

            # Calculate IoU matrix
            iou_matrix = bbox_iou(det_preds, boxes)
            # Determine best match for each prediction
            matched_indices = torch.argmax(iou_matrix, dim=1)
            max_ious = iou_matrix[range(len(det_preds)), matched_indices]
            # Apply IoU threshold to determine positive matches
            matched_gt_indices = matched_indices[max_ious > 0.5]
            unmatched_indices = matched_indices[max_ious <= 0.5]

            # Set confidence targets based on matching
            target_confidences = torch.zeros_like(conf_preds[:, 0])
            matched_gt_indices = matched_indices[matched_indices != -1]
            unmatched_gt_indices = matched_indices[matched_indices == -1]

            target_confidences[matched_gt_indices] = 1

            # Classification labels - background class for unmatched, actual labels for matched
            cls_labels = torch.full((len(cls_preds),), background_class_label, device=device, dtype=torch.long)

            #print(matched_gt_indices.shape, labels.shape)


            cls_labels[matched_gt_indices] = labels[matched_indices[matched_indices != -1]]

            # Compute losses
            regression_loss = iou_loss(det_preds[matched_gt_indices], boxes[matched_gt_indices])
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
