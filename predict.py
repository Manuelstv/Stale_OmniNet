import torch
import cv2
import os
import numpy as np
#from torchvision.transforms import transforms
from model import SimpleObjectDetector
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
model = SimpleObjectDetector(num_boxes=50, num_classes=38).to(device)
model.load_state_dict(torch.load('best.pth'))  # Load the saved weights
model.eval()  # Set the model to evaluation mode

# Process the image
image_path = 'images/image_00295.jpg'  # Replace with your image path
image = process_image(image_path)

# Predict
with torch.no_grad():
    detection_preds, classification_preds, confidence_preds = model(image)

def draw_boxes(image, boxes, labels, confidences, label_names, threshold=0.05):
    for box, label, confidence in zip(boxes, labels, confidences):
        if confidence < threshold:
            continue
        label_name = label_names[label]

        x_center, y_center, width, height = 1920*box[0], 960*box[1], 20*box[2], 20*box[3]
        xmin = int((x_center - width / 2))
        ymin = int(y_center - height / 2) 
        xmax = int(x_center + width / 2) 
        ymax = int((y_center + height / 2))


        print(box)
        # Draw the rectangle
        cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
        cv2.putText(image, f"{label_name}: {confidence:.2f}", (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)


# Convert predictions to numpy arrays for easier processing
detection_preds = detection_preds.cpu().numpy()[0]
classification_preds = torch.softmax(classification_preds, dim=-1)
classification_preds = classification_preds.cpu().numpy()[0]

# Define label names (adjust according to your dataset)
label_mapping = {'t':0, 'airconditioner': 1, 'backpack': 2, 'bathtub': 3, 'bed': 4, 'board': 5, 'book': 6, 'bottle': 7, 'bowl': 8, 'cabinet': 9, 'chair': 10, 'clock': 11, 'computer': 12, 'cup': 13, 'door': 14, 'fan': 15, 'fireplace': 16, 'heater': 17, 'keyboard': 18, 'light': 19, 'microwave': 20, 'mirror': 21, 'mouse': 22, 'oven': 23, 'person': 24, 'phone': 25, 'picture': 26, 'potted plant': 27, 'refrigerator': 28, 'sink': 29, 'sofa': 30, 'table': 31, 'toilet': 32, 'tv': 33, 'vase': 34, 'washer': 35, 'window': 36, 'wine glass': 37}

label_names = {v: k for k, v in label_mapping.items()}


image = image.cpu().numpy()
image = np.squeeze(image)
image = np.transpose(image, (1, 2, 0))

for i in range(detection_preds.shape[0]):
    box = detection_preds[i, :4]
    print(box)
    label = np.argmax(classification_preds[i])
    confidence = classification_preds[i][label]

    # Assuming box format is [xmin, ymin, xmax, ymax]
    draw_boxes(image, [box], [label], [confidence], label_names)

# Display the image or save it
cv2.imshow("Detected Image", image)
cv2.waitKey(0)  # Wait for a key press to close the displayed image
cv2.destroyAllWindows()