import torch
from foviou import *
from scipy.optimize import linear_sum_assignment
import torch.nn.functional as F


def hungarian_matching(gt_boxes_in, pred_boxes_in, new_w, new_h):
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
    gt_boxes = gt_boxes_in.clone().to(torch.float)
    pred_boxes = pred_boxes_in.clone().to(torch.float)

    iou_matrix = fov_iou_batch(gt_boxes, pred_boxes, new_w, new_h)

    # Convert IoUs to cost
    cost_matrix = 1 - iou_matrix.detach().cpu().numpy()

    # Apply Hungarian matching
    gt_indices, pred_indices = linear_sum_assignment(cost_matrix)

    # Extract the matched pairs
    matched_pairs = [(gt_indices[i], pred_indices[i]) for i in range(len(gt_indices))]


    return matched_pairs, iou_matrix

def custom_loss_function(det_preds, conf_preds, boxes, new_w, new_h):
    iou_threshold = 0.5  # Define your IoU threshold
    matches, _ = hungarian_matching(boxes, det_preds, new_w, new_h)
    
    total_loss = 0.0
    total_confidence_loss = 0.0
    total_localization_loss = 0.0

    for gt_idx, pred_idx in matches:
        gt_box = boxes[gt_idx]
        pred_box = det_preds[pred_idx]
        pred_confidence = conf_preds[pred_idx]

        # Ensure pred_confidence is a single value tensor
        pred_confidence = pred_confidence.view(-1)  # Reshape to ensure it's a 1D tensor

        # Compute the IoU for this pair
        iou = fov_iou(pred_box, gt_box)

        # Define target confidence based on IoU
        target_confidence = torch.tensor([1 if iou > iou_threshold else 0], dtype=torch.float, device=pred_confidence.device)
        
        # Compute the confidence loss (BCE)
        confidence_loss = F.binary_cross_entropy(pred_confidence, target_confidence)
        total_confidence_loss += confidence_loss

        # Compute the localization loss for this pair (1 - IoU or MSE)
        localization_loss = 1 - iou  # or F.mse_loss(pred_box, gt_box)
        total_localization_loss += localization_loss

    # Calculate the mean losses
    mean_confidence_loss = total_confidence_loss / len(matches) if matches else torch.tensor(0.0)
    mean_localization_loss = total_localization_loss / len(matches) if matches else torch.tensor(0.0)

    # Combine confidence loss and localization loss
    total_loss = mean_confidence_loss + mean_localization_loss

    return total_loss


def giou_loss(pred_boxes_in, gt_boxes_in, new_w, new_h):
    # Ensure the boxes are (x_min, y_min, x_max, y_max)
    assert pred_boxes_in.shape == gt_boxes_in.shape

    pred_boxes_in = pred_boxes_in.unsqueeze(0)
    gt_boxes_in = gt_boxes_in.unsqueeze(0)
    
    pred_boxes = pred_boxes_in.clone()
    gt_boxes = gt_boxes_in.clone()

    gt_boxes[:, 0] = gt_boxes_in[:, 0]*new_w
    gt_boxes[:, 1] = gt_boxes_in[:, 1]*new_h
    gt_boxes[:, 2] = gt_boxes_in[:, 2]*new_w
    gt_boxes[:, 3] = gt_boxes_in[:, 3]*new_h

    gt_boxes = gt_boxes.to(torch.int)

    pred_boxes[:, 0] = int(pred_boxes_in[:, 0]*new_w)
    pred_boxes[:, 1] = int(pred_boxes_in[:, 1]*new_h)
    pred_boxes[:, 2] = int(pred_boxes_in[:, 2]*new_w)
    pred_boxes[:, 3] = int(pred_boxes_in[:, 3]*new_h)

    #pred_boxes = pred_boxes.to(torch.int)

    # Intersection
    inter_xmin = torch.max(pred_boxes[:, 0], gt_boxes[:, 0])
    inter_ymin = torch.max(pred_boxes[:, 1], gt_boxes[:, 1])
    inter_xmax = torch.min(pred_boxes[:, 2], gt_boxes[:, 2])
    inter_ymax = torch.min(pred_boxes[:, 3], gt_boxes[:, 3])
    
    inter_area = torch.clamp(inter_xmax - inter_xmin, min=0) * torch.clamp(inter_ymax - inter_ymin, min=0)

    # Union
    pred_area = (pred_boxes[:, 2] - pred_boxes[:, 0]) * (pred_boxes[:, 3] - pred_boxes[:, 1])
    gt_area = (gt_boxes[:, 2] - gt_boxes[:, 0]) * (gt_boxes[:, 3] - gt_boxes[:, 1])
    union_area = pred_area + gt_area - inter_area

    # Enclosing box
    enc_xmin = torch.min(pred_boxes[:, 0], gt_boxes[:, 0])
    enc_ymin = torch.min(pred_boxes[:, 1], gt_boxes[:, 1])
    enc_xmax = torch.max(pred_boxes[:, 2], gt_boxes[:, 2])
    enc_ymax = torch.max(pred_boxes[:, 3], gt_boxes[:, 3])
    enc_area = (enc_xmax - enc_xmin) * (enc_ymax - enc_ymin)

    # IoU and GIoU
    iou = inter_area / union_area
    giou = iou - (enc_area - union_area) / enc_area
    giou_loss = 1 - giou  # GIoU loss

    return giou_loss.mean()

def diou_loss(pred_boxes_in, gt_boxes_in, new_w, new_h):
    # Ensure the boxes are (x_min, y_min, x_max, y_max)
    assert pred_boxes_in.shape == gt_boxes_in.shape

    pred_boxes_in = pred_boxes_in.unsqueeze(0)
    gt_boxes_in = gt_boxes_in.unsqueeze(0)
    
    pred_boxes = pred_boxes_in.clone()
    gt_boxes = gt_boxes_in.clone()

    gt_boxes[:, 0] = gt_boxes_in[:, 0]*new_w
    gt_boxes[:, 1] = gt_boxes_in[:, 1]*new_h
    gt_boxes[:, 2] = gt_boxes_in[:, 2]*new_w
    gt_boxes[:, 3] = gt_boxes_in[:, 3]*new_h

    gt_boxes = gt_boxes.to(torch.int)

    pred_boxes[:, 0] = int(pred_boxes_in[:, 0]*new_w)
    pred_boxes[:, 1] = int(pred_boxes_in[:, 1]*new_h)
    pred_boxes[:, 2] = int(pred_boxes_in[:, 2]*new_w)
    pred_boxes[:, 3] = int(pred_boxes_in[:, 3]*new_h)

    #pred_boxes = pred_boxes.to(torch.int)

    # Intersection
    inter_xmin = torch.max(pred_boxes[:, 0], gt_boxes[:, 0])
    inter_ymin = torch.max(pred_boxes[:, 1], gt_boxes[:, 1])
    inter_xmax = torch.min(pred_boxes[:, 2], gt_boxes[:, 2])
    inter_ymax = torch.min(pred_boxes[:, 3], gt_boxes[:, 3])
    
    inter_area = torch.clamp(inter_xmax - inter_xmin, min=0) * torch.clamp(inter_ymax - inter_ymin, min=0)

    # Union
    pred_area = (pred_boxes[:, 2] - pred_boxes[:, 0]) * (pred_boxes[:, 3] - pred_boxes[:, 1])
    gt_area = (gt_boxes[:, 2] - gt_boxes[:, 0]) * (gt_boxes[:, 3] - gt_boxes[:, 1])
    union_area = pred_area + gt_area - inter_area

    # Enclosing box
    enc_xmin = torch.min(pred_boxes[:, 0], gt_boxes[:, 0])
    enc_ymin = torch.min(pred_boxes[:, 1], gt_boxes[:, 1])
    enc_xmax = torch.max(pred_boxes[:, 2], gt_boxes[:, 2])
    enc_ymax = torch.max(pred_boxes[:, 3], gt_boxes[:, 3])
    enc_area = (enc_xmax - enc_xmin) * (enc_ymax - enc_ymin)

    # IoU and GIoU
    iou = inter_area / union_area

    pred_boxes_center = (pred_boxes[:, 2:4] + pred_boxes[:, 0:2]) / 2
    gt_boxes_center = (gt_boxes[:, 2:4] + gt_boxes[:, 0:2]) / 2

    # Calculate the euclidean distance between centers
    center_distance = torch.norm(pred_boxes_center - gt_boxes_center, dim=1)

    # Calculate the diagonal length of the smallest enclosing box
    enc_diag = torch.norm(torch.tensor([enc_xmax - enc_xmin, enc_ymax - enc_ymin], device=pred_boxes.device), dim=0)

    # DIoU
    diou = iou - (center_distance ** 2) / (enc_diag ** 2)
    diou_loss = 1 - diou

    return diou_loss.mean()