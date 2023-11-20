import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
#sfrom torchvision import transforms

# Import your model and dataset classes
from model import SimpleObjectDetector
from datasets import PascalVOCDataset
torch.cuda.empty_cache()

def bbox_iou(box1, box2):
    """
    Calculate the IoU of two bounding boxes.
    """
    # Get the coordinates of bounding boxes
    b1_x1, b1_y1, b1_x2, b1_y2 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
    b2_x1, b2_y1, b2_x2, b2_y2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]

    # Get the coordinates of the intersection rectangle
    inter_rect_x1 = torch.max(b1_x1, b2_x1)
    inter_rect_y1 = torch.max(b1_y1, b2_y1)
    inter_rect_x2 = torch.min(b1_x2, b2_x2)
    inter_rect_y2 = torch.min(b1_y2, b2_y2)

    # Intersection area
    inter_area = torch.clamp(inter_rect_x2 - inter_rect_x1 + 1, min=0) * torch.clamp(inter_rect_y2 - inter_rect_y1 + 1, min=0)

    # Union Area
    b1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1)
    b2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1)

    iou = inter_area / (b1_area + b2_area - inter_area)

    return iou

def iou_loss(preds, targets):
    """
    Calculate the IoU loss for the predicted and target bounding boxes.
    """
    iou = bbox_iou(preds, targets)
    loss = 1 - iou  # IoU loss is 1 - IoU
    return loss.mean()

torch.cuda.empty_cache()
# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
num_epochs = 60
learning_rate = 0.00001
batch_size = 8
num_classes = 37  # Example: 20 classes

# Initialize dataset and dataloader
train_dataset = PascalVOCDataset(split='TRAIN', keep_difficult=False, max_images=800)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=train_dataset.collate_fn)

model = SimpleObjectDetector(num_boxes=80, num_classes=num_classes).to(device)

# Loss and optimizer
classification_criterion = nn.CrossEntropyLoss()  # For class predictions
confidence_criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

x = 50  # Define the interval for printing the loss

for epoch in range(num_epochs):
    torch.cuda.empty_cache()
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
            labels = labels - 1

            target_confidences = torch.zeros_like(conf_preds)
            target_confidences[:len(boxes)] = 1

            # Match the number of predictions to the number of ground truth boxes
            num_ground_truth = boxes.shape[0]
            det_preds = det_preds[:num_ground_truth, :]
            cls_preds = cls_preds[:num_ground_truth, :].view(-1, num_classes)
            #conf_preds = conf_preds[:num_ground_truth, :]

            # Compute loss for this image
            confidence_loss = confidence_criterion(conf_preds, target_confidences)

            # Combine losses
            regression_loss = iou_loss(det_preds, boxes)
            classification_loss = classification_criterion(cls_preds, labels)

            # Accumulate losses
            total_regression_loss += regression_loss
            total_confidence_loss += confidence_loss
            total_classification_loss += classification_loss

        # Compute the average losses
        avg_regression_loss = total_regression_loss / len(boxes_list)
        avg_classification_loss = total_classification_loss / len(boxes_list)
        avg_confidence_loss = total_confidence_loss / len(boxes_list)
        
        total_loss = avg_regression_loss + avg_classification_loss + avg_confidence_loss

        # Backward pass and optimization
        total_loss.backward()
        optimizer.step()

        # Print the loss every x iterations
        if (i + 1) % x == 0:
            print(f"Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{len(train_loader)}], Loss: {total_loss.item()}")

model_file = f"best.pth"
torch.save(model.state_dict(), model_file)
print(f"Model saved to {model_file} after Epoch {epoch + 1}")

print('Training completed.')
