import torch
import cv2
import os
import numpy as np
#from torchvision.transforms import transforms
from model import SimpleObjectDetectorWithBackbone
from datasets import PascalVOCDataset
from utils import transform

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Function to load an image and transform it
def process_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.resize(image, (600,300))  # Resize the image
    image = image.astype(np.float32) / 255.0
    image = torch.from_numpy(image).permute(2, 0, 1)  # Convert to torch tensor and permute to (C, H, W)
    return image.unsqueeze(0).to(device)  # Add batch dimension

# Load your trained model
model = SimpleObjectDetectorWithBackbone(num_boxes=30, num_classes=37).to(device)
model.load_state_dict(torch.load('best.pth'))  # Load the saved weights
model.eval()  # Set the model to evaluation mode

# Process the image
image_path = 'images/image_00297.jpg'  # Replace with your image path
image = process_image(image_path)

# Predict
with torch.no_grad():
    detection_preds, classification_preds, confidence_preds = model(image)

def draw_boxes(image, boxes, labels, confidences, label_names, threshold=0.03):
    for box, label, confidence in zip(boxes, labels, confidences):
        if confidence < threshold:
            continue
        label_name = label_names[label]

        x_center, y_center, width, height = 1920*box[0], 960*box[1], 20*box[2], 20*box[3]
        xmin = int((x_center - width / 2))
        ymin = int(y_center - height / 2) 
        xmax = int(x_center + width / 2) 
        ymax = int((y_center + height / 2))

        # Draw the rectangle
        cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
        cv2.putText(image, f"{label_name}: {confidence:.2f}", (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)


# Convert predictions to numpy arrays for easier processing
detection_preds = detection_preds.cpu().numpy()[0]
classification_preds = torch.softmax(classification_preds, dim=-1)
classification_preds = classification_preds.cpu().numpy()[0]

# Define label names (adjust according to your dataset)
label_mapping ={
    'airconditioner': 0,
    'backpack': 1,
    'bathtub': 2,
    'bed': 3,
    'board': 4,
    'book': 5,
    'bottle': 6,
    'bowl': 7,
    'cabinet': 8,
    'chair': 9,
    'clock': 10,
    'computer': 11,
    'cup': 12,
    'door': 13,
    'fan': 14,
    'fireplace': 15,
    'heater': 16,
    'keyboard': 17,
    'light': 18,
    'microwave': 19,
    'mirror': 20,
    'mouse': 21,
    'oven': 22,
    'person': 23,
    'phone': 24,
    'picture': 25,
    'potted plant': 26,
    'refrigerator': 27,
    'sink': 28,
    'sofa': 29,
    'table': 30,
    'toilet': 31,
    'tv': 32,
    'vase': 33,
    'washer': 34,
    'window': 35,
    'wine glass': 36}

label_names = {v: k for k, v in label_mapping.items()}


image = image.cpu().numpy()
image = np.squeeze(image)
image = np.transpose(image, (1, 2, 0))

for i in range(detection_preds.shape[0]):
    box = detection_preds[i, :4]
    print(box)
    label = np.argmax(classification_preds[i])
    confidence = classification_preds[i][label]

    draw_boxes(image, [box], [label], [confidence], label_names)

# Display the image or save it
cv2.imshow("Detected Image", image)
cv2.waitKey(0)  # Wait for a key press to close the displayed image
cv2.destroyAllWindows()