import torch
from foviou import *
from scipy.optimize import linear_sum_assignment
import torch.nn.functional as F
import pdb


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
    iou_threshold = 0.2  # Define your IoU threshold
    matches, _ = hungarian_matching(boxes, det_preds, new_w, new_h)

    total_loss = 0.0
    total_confidence_loss = 0.0
    total_localization_loss = 0.0
    unmatched_loss = 0.0  # Initialize the unmatched loss

    matched_dets = set(pred_idx for _, pred_idx in matches)
    all_dets = set(range(len(det_preds)))
    unmatched_dets = all_dets - matched_dets

    for gt_idx, pred_idx in matches:
        gt_box = boxes[gt_idx]
        pred_box = det_preds[pred_idx]
        pred_confidence = conf_preds[pred_idx]

        pred_confidence = pred_confidence.view(-1)  # Reshape to ensure it's a 1D tensor

        iou = fov_iou(pred_box, gt_box)
        target_confidence = torch.tensor([1 if iou > iou_threshold else 0], dtype=torch.float, device=pred_confidence.device)
        
        confidence_loss = F.binary_cross_entropy(pred_confidence, target_confidence)
        total_confidence_loss += confidence_loss

        localization_loss = 1 - iou  # or another loss like F.mse_loss(pred_box, gt_box)
        total_localization_loss += localization_loss

    # Penalty for each unmatched detection
    unmatched_penalty = 0.5  # Define your penalty for unmatched detections
    for det_idx in unmatched_dets:
        unmatched_confidence = conf_preds[det_idx]
        unmatched_confidence = unmatched_confidence.view(-1)  # Reshape to ensure it's a 1D tensor
        unmatched_loss += F.binary_cross_entropy(unmatched_confidence, torch.tensor([0.0], dtype=torch.float, device=unmatched_confidence.device))

    total_loss = (total_confidence_loss + total_localization_loss + unmatched_penalty * unmatched_loss) / (len(matches) + len(unmatched_dets)) if matches else unmatched_penalty * unmatched_loss

    return total_loss, matches





def custom_loss_function2(det_preds, conf_preds, boxes, new_w, new_h):
    iou_threshold = 0.3  # Define your IoU threshold
    matches, _ = hungarian_matching(boxes, det_preds, new_w, new_h)

    #print(matches)
    
    total_loss = 0.0
    total_confidence_loss = 0.0
    total_localization_loss = 0.0

    #should add loss term for no matches

    for gt_idx, pred_idx in matches:
        gt_box = boxes[gt_idx]
        pred_box = det_preds[pred_idx]
        pred_confidence = conf_preds[pred_idx]

        # Ensure pred_confidence is a single value tensor
        pred_confidence = pred_confidence.view(-1)  # Reshape to ensure it's a 1D tensor

        # Compute the IoU for this pair
        iou = fov_iou(pred_box, gt_box)

        # Define target confidence based on IoU
        #pdb.set_trace()
        target_confidence = torch.tensor([1 if iou > iou_threshold else 0], dtype=torch.float, device=pred_confidence.device)
        
        # Compute the confidence loss (BCE)
        confidence_loss = F.binary_cross_entropy(pred_confidence, target_confidence)
        total_confidence_loss += confidence_loss

        # Compute the localization loss for this pair (1 - IoU or MSE)
        localization_loss = 1 - iou  # or F.mse_loss(pred_box, gt_box)
        #localization_loss = fov_giou_loss(pred_box, gt_box)
        total_localization_loss += localization_loss

    # Calculate the mean losses
    mean_confidence_loss = total_confidence_loss / len(matches) if matches else torch.tensor(0.0)
    mean_localization_loss = total_localization_loss / len(matches) if matches else torch.tensor(0.0)

    # Combine confidence loss and localization loss
    total_loss = 0.5*mean_confidence_loss + 0.5*mean_localization_loss

    return total_loss, matches
