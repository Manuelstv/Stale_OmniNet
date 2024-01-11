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

if __name__ == "__main__":

    #torch.cuda.empty_cache()

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Hyperparameters
    num_epochs = 5
    learning_rate = 0.00001
    batch_size = 10
    num_classes = 3
    max_images = 10
    num_boxes = 3

    new_w, new_h = 600,300


    # Initialize dataset and dataloader
    train_dataset = PascalVOCDataset(split='TRAIN', keep_difficult=False, max_images=max_images)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=train_dataset.collate_fn)

    model = SimpleObjectDetector(num_boxes=num_boxes, num_classes=num_classes).to(device)
    model.fc1.apply(init_weights)
    model.det_head.apply(init_weights)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        total_matches = 0
        total_regression_loss = torch.tensor(0.0, device=device, requires_grad=True)

        for i, (images, boxes_list, labels_list, confidences_list) in enumerate(train_loader):
            images = images.to(device)
            optimizer.zero_grad()
            detection_preds = model(images)
            losses = []
            n = 0

            for boxes, labels, det_preds in zip(boxes_list, labels_list, detection_preds):
                boxes = boxes.to(device)
                det_preds = det_preds.to(device)     
                labels = labels.to(device)
                matches, matched_iou_scores = hungarian_matching(boxes, det_preds)
                regression_loss = (1 - matched_iou_scores).mean()
                losses.append(regression_loss)
                total_matches += len(matches)
                
                #salvando images de 10 em 10 épocas
                if epoch>0 and epoch%100==0:
                #if True:
                    #pass
                    img1 = images[n].mul(255).clamp(0, 255).permute(1, 2, 0).cpu().numpy().astype(np.uint8).copy()
                    #cv2.imwrite('branco2.jpg', img1)

                    for box in boxes:
                        x_min, y_min, x_max, y_max = int(box[0]*new_w), int(box[1]*new_h), int(box[2]*new_w), int(box[3]*new_h)
                        #images = np.ascontiguousarray(images, dtype = np.uint8)
                        cv2.rectangle(img1, (x_min, y_min), (x_max, y_max), (0,255,0))

                    for box in det_preds:
                        x_min, y_min, x_max, y_max = int(box[0]*new_w), int(box[1]*new_h), int(box[2]*new_w), int(box[3]*new_h)
                        #images = np.ascontiguousarray(images, dtype = np.uint8)
                        cv2.rectangle(img1, (x_min, y_min), (x_max, y_max), (255,0,0))
                    cv2.imwrite(f'/home/mstveras/images/img{n}.jpg', img1)

                    #img = process_and_save_image_planar(img1, boxes.cpu().detach().numpy(), (0,255,0), f'/home/mstveras/images/img{n}.jpg')
                    #img = process_and_save_image_planar(img, det_preds.cpu().detach().numpy(), (255,0,0), f'/home/mstveras/images/img2_{n}.jpg')
                
                    n+=1

            if total_matches > 0:
                avg_regression_loss = sum(losses) / total_matches
                avg_regression_loss.backward()  # Backpropagate here
                optimizer.step()  # Update the weights once per batch
                total_loss += avg_regression_loss.item()
            else:
                print('No matches found, not backpropagating.')

        avg_epoch_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch}/{num_epochs}: Loss: {avg_epoch_loss}")

    model_file = "best.pth"
    torch.save(model.state_dict(), model_file)
    print(f"Model saved to {model_file} after Epoch {epoch + 1}")
    print('Training completed.')