import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleObjectDetector(nn.Module):
    def __init__(self, num_boxes=50, num_classes=37):
        super(SimpleObjectDetector, self).__init__()
        self.num_boxes = num_boxes
        self.num_classes = num_classes

        # Original convolutional layers
        self.conv1 = nn.Conv2d(3, 8, kernel_size=3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(8)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(16)
        self.conv3 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(32)

        # Additional convolutional layers
        self.conv4 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(32)
        self.conv5 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.bn5 = nn.BatchNorm2d(32)

        # Pooling layer
        self.pool = nn.MaxPool2d(2, 2)

        # Calculate the flattened size after convolution and pooling layers
        # Assuming the input image size is 1920x960
        # The size after conv and pool layers would still be [32, 120, 60]
        fc_input_features = 32 * 120 * 60

        # Fully connected layers
        self.fc1 = nn.Linear(fc_input_features, 256)
        self.det_head = nn.Linear(256, num_boxes * 4)  # Detection head
        self.cls_head = nn.Linear(256, num_boxes * num_classes)  # Classification head
        self.conf_head = nn.Linear(256, num_boxes)  # Confidence head

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(x)))
        x = self.pool(x)

        # Flatten the features for the fully connected layer
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))

        # Apply detection, classification, and confidence heads
        detection = self.det_head(x).view(-1, self.num_boxes, 4)
        classification = self.cls_head(x).view(-1, self.num_boxes, self.num_classes)
        confidence = torch.sigmoid(self.conf_head(x)).view(-1, self.num_boxes, 1)

        return detection, classification, confidence