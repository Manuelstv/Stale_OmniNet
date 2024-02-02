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


def custom_loss_function3(det_preds, conf_preds, boxes, labels, class_preds, new_w, new_h):
    #alta sensibilidade a esse param
    iou_threshold = 0.2  # Define your IoU threshold
    matches, _ = hungarian_matching(boxes, det_preds, new_w, new_h)

    total_loss = 0.0
    total_confidence_loss = 0.0
    total_localization_loss = 0.0
    total_classification_loss = 0.0
    unmatched_loss = 0.0  # Initialize the unmatched loss

    matched_dets = set(pred_idx for _, pred_idx in matches)
    all_dets = set(range(len(det_preds)))
    unmatched_dets = all_dets - matched_dets

    for gt_idx, pred_idx in matches:
        gt_box = boxes[gt_idx]
        pred_box = det_preds[pred_idx]
        pred_confidence = conf_preds[pred_idx]
        class_label = labels[gt_idx]
        pred_class = class_preds[pred_idx]

        pred_confidence = pred_confidence.view(-1)  # Reshape to ensure it's a 1D tensor

        iou = fov_iou(pred_box, gt_box)
        target_confidence = torch.tensor([1 if iou > iou_threshold else 0], dtype=torch.float, device=pred_confidence.device)
        
        confidence_loss = F.binary_cross_entropy(pred_confidence, target_confidence)
        total_confidence_loss += confidence_loss

        class_criterion = torch.nn.CrossEntropyLoss()
        class_label = class_label.unsqueeze(0)  # Reshape to [1]
        pred_class = pred_class.unsqueeze(0) 

        classification_loss = class_criterion(pred_class, class_label)

        total_classification_loss += classification_loss

        localization_loss = 1 - iou  # or another loss like F.mse_loss(pred_box, gt_box)
        total_localization_loss += localization_loss

    # Penalty for each unmatched detection
    unmatched_penalty = 0.5  # Define your penalty for unmatched detections
    for det_idx in unmatched_dets:
        unmatched_confidence = conf_preds[det_idx]
        unmatched_confidence = unmatched_confidence.view(-1)  # Reshape to ensure it's a 1D tensor
        unmatched_loss += F.binary_cross_entropy(unmatched_confidence, torch.tensor([0.0], dtype=torch.float, device=unmatched_confidence.device))

    #print(total_classification_loss*0.1)
    #print(total_confidence_loss)
    #print(total_localization_loss)

    total_loss = (total_confidence_loss + total_localization_loss+total_classification_loss*0.1+unmatched_penalty * unmatched_loss) / (len(matches) + len(unmatched_dets)) if matches else unmatched_penalty * unmatched_loss

    return total_loss, matches


def custom_loss_function(det_preds, conf_preds, boxes, labels, class_preds, new_w, new_h):
    """
    Calculate the custom loss for an object detection model.

    This function computes a composite loss that includes confidence loss, 
    localization loss, classification loss, and a penalty for unmatched detections.
    Localization loss is applied only to detections with confidence above a 
    specified threshold, aligning this approach closer to Faster R-CNN's methodology.

    Parameters:
    - det_preds (Tensor): The predicted bounding boxes of shape (N, 4), where N is 
                          the number of detections, and each bounding box is 
                          represented as (x1, y1, x2, y2).
    - conf_preds (Tensor): The confidence scores for each predicted bounding box, 
                           of shape (N,).
    - boxes (Tensor): The ground truth bounding boxes of shape (M, 4), where M is 
                      the number of ground truth objects.
    - labels (Tensor): The ground truth labels for each object, of shape (M,).
    - class_preds (Tensor): The class predictions for each detected bounding box, 
                            of shape (N, num_classes).
    - new_w (int/float): The width scaling factor for coordinate normalization.
    - new_h (int/float): The height scaling factor for coordinate normalization.

    Returns:
    - total_loss (Tensor): The computed total loss as a scalar tensor.
    - matches (list of tuples): List of matched ground truth and prediction 
                                indices as (ground_truth_idx, prediction_idx).

    The loss computation involves the following steps:
    - Matching predicted and ground truth boxes using the Hungarian algorithm.
    - Computing the confidence loss using binary cross-entropy.
    - Computing the localization loss (1 - IoU) only for predictions with 
      confidence above a threshold (0.5 by default).
    - Computing the classification loss using cross-entropy.
    - Adding a penalty for each unmatched detection.

    Note:
    - The function assumes that the bounding box coordinates are normalized 
      using the provided scaling factors (new_w, new_h).
    - The IoU threshold for determining positive matches is set to 0.4.
    - The unmatched penalty is set to 2.5.
    - The loss components are normalized by the total number of matches and 
      unmatched detections.
    """
    
    iou_threshold = 0.3  # IoU threshold
    confidence_threshold = 0.3  # Confidence threshold for applying regression loss
    matches, _ = hungarian_matching(boxes, det_preds, new_w, new_h)

    total_loss = 0.0
    total_confidence_loss = 0.0
    total_localization_loss = 0.0
    total_classification_loss = 0.0
    unmatched_loss = 0.0

    matched_dets = set(pred_idx for _, pred_idx in matches)
    all_dets = set(range(len(det_preds)))
    unmatched_dets = all_dets - matched_dets

    for gt_idx, pred_idx in matches:
        gt_box = boxes[gt_idx]
        pred_box = det_preds[pred_idx]
        pred_confidence = conf_preds[pred_idx].view(-1)
        class_label = labels[gt_idx]
        pred_class = class_preds[pred_idx]

        iou = fov_iou(pred_box, gt_box)
        target_confidence = torch.tensor([1.0 if iou > iou_threshold else 0], dtype=torch.float, device=pred_confidence.device)
        confidence_loss = F.binary_cross_entropy(pred_confidence, target_confidence)
        total_confidence_loss += confidence_loss

        class_criterion = torch.nn.CrossEntropyLoss()
        classification_loss = class_criterion(pred_class.unsqueeze(0), class_label.unsqueeze(0))
        total_classification_loss += classification_loss

        # Apply localization loss only for confident predictions
        if pred_confidence.item() > confidence_threshold:
            localization_loss = 1 - iou
            total_localization_loss += localization_loss

    # Penalty for each unmatched detection
    unmatched_penalty = 0.5
    for det_idx in unmatched_dets:
        unmatched_confidence = conf_preds[det_idx].view(-1)
        unmatched_loss += F.binary_cross_entropy(unmatched_confidence, torch.tensor([0.0], dtype=torch.float, device=unmatched_confidence.device))

    total_loss = (total_confidence_loss + total_localization_loss + 0.1*total_classification_loss + unmatched_penalty * unmatched_loss) / (len(matches) + len(unmatched_dets)) if matches else unmatched_penalty * unmatched_loss * 5

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
