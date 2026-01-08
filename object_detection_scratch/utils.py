import torch
import torch.nn.functional as F
import torchvision.ops as ops
import math

class BoxCoder:
    """
    Encodes/Decodes bounding boxes for training RPN and Fast R-CNN heads.
    """
    def __init__(self, weights=(1.0, 1.0, 1.0, 1.0)):
        self.weights = weights

    def encode(self, matched_gt_boxes, proposals):
        """
        Compute regression targets (dx, dy, dw, dh) relative to proposals.
        """
        # proposals: [N, 4], matched_gt_boxes: [N, 4]
        wx, wy, ww, wh = self.weights
        
        ex, ey, ew, eh = self._get_xywh(proposals)
        gt_x, gt_y, gt_w, gt_h = self._get_xywh(matched_gt_boxes)
        
        dx = wx * (gt_x - ex) / ew
        dy = wy * (gt_y - ey) / eh
        dw = ww * torch.log(gt_w / ew)
        dh = wh * torch.log(gt_h / eh)
        
        targets = torch.stack((dx, dy, dw, dh), dim=-1)
        return targets

    def decode(self, rel_codes, boxes):
        """
        Apply regression targets to boxes to get predicted boxes.
        """
        boxes = boxes.to(rel_codes.dtype)

        wx, wy, ww, wh = self.weights
        dx = rel_codes[..., 0] / wx
        dy = rel_codes[..., 1] / wy
        dw = rel_codes[..., 2] / ww
        dh = rel_codes[..., 3] / wh

        ex, ey, ew, eh = self._get_xywh(boxes)
        # Prevent sending too large values into exp
        dw = torch.clamp(dw, max=math.log(1000.0 / 16))
        dh = torch.clamp(dh, max=math.log(1000.0 / 16))

        pred_ctr_x = dx * ew + ex
        pred_ctr_y = dy * eh + ey
        pred_w = torch.exp(dw) * ew
        pred_h = torch.exp(dh) * eh

        pred_boxes1 = pred_ctr_x - 0.5 * pred_w
        pred_boxes2 = pred_ctr_y - 0.5 * pred_h
        pred_boxes3 = pred_ctr_x + 0.5 * pred_w
        pred_boxes4 = pred_ctr_y + 0.5 * pred_h
        
        pred_boxes = torch.stack((pred_boxes1, pred_boxes2, pred_boxes3, pred_boxes4), dim=-1)
        return pred_boxes

    def _get_xywh(self, boxes):
        # x1, y1, x2, y2 -> ctr_x, ctr_y, w, h
        w = boxes[..., 2] - boxes[..., 0]
        h = boxes[..., 3] - boxes[..., 1]
        ctr_x = boxes[..., 0] + 0.5 * w
        ctr_y = boxes[..., 1] + 0.5 * h
        return ctr_x, ctr_y, w, h

class Matcher:
    """
    Matches proposals to ground truth boxes based on IoU.
    """
    def __init__(self, high_threshold, low_threshold, allow_low_quality_matches=False):
        self.high_threshold = high_threshold
        self.low_threshold = low_threshold
        self.allow_low_quality_matches = allow_low_quality_matches

    def __call__(self, match_quality_matrix):
        """
        Returns:
            matches: Tensor of shape [M] (M = #proposals)
                Values are:
                >= 0: index of matched GT
                -1: negative (below low_threshold)
                -2: ignore (between low and high, or specific ignore/neutral region)
        """
        # matrix: [N_gt, N_proposals] -> IoU
        # For each proposal, find best GT
        matched_vals, matches = match_quality_matrix.max(dim=0) # [N_prop], [N_prop]
        
        # Initialize with -1 (negative)
        labels = torch.full_like(matches, -1)
        
        # Positives
        labels[matched_vals >= self.high_threshold] = matches[matched_vals >= self.high_threshold]
        
        # Low quality matches (optional) - e.g. for RPN ensure every GT has at least one anchor??
        if self.allow_low_quality_matches:
            # Find best proposal for each GT
            highest_quality_foreach_gt, _ = match_quality_matrix.max(dim=1) # [N_gt]
            # Find all proposals that share this max quality (could be ties)
            gt_pred_pairs_of_highest_quality = torch.where(
                match_quality_matrix == highest_quality_foreach_gt[:, None]
            )
            pred_inds_to_update = gt_pred_pairs_of_highest_quality[1]
            labels[pred_inds_to_update] = matches[pred_inds_to_update]

        # Negatives are already -1? No, we need to handle "between" encoded as -2
        # Logic: 
        # < low_threshold -> -1
        # >= low_threshold && < high_threshold -> -2
        # >= high_threshold -> index
        
        # Set "ignore" first
        ignore_mask = (matched_vals >= self.low_threshold) & (matched_vals < self.high_threshold)
        labels[ignore_mask] = -2 
        
        return labels

def box_iou(boxes1, boxes2):
    return ops.box_iou(boxes1, boxes2)

def nms(boxes, scores, iou_threshold):
    return ops.nms(boxes, scores, iou_threshold)

def clip_boxes_to_image(boxes, size):
    """
    Clip boxes so that they lie inside an image of size (height, width).
    """
    h, w = size
    boxes_x1 = boxes[:, 0::4].clamp(min=0, max=w)
    boxes_y1 = boxes[:, 1::4].clamp(min=0, max=h)
    boxes_x2 = boxes[:, 2::4].clamp(min=0, max=w)
    boxes_y2 = boxes[:, 3::4].clamp(min=0, max=h)
    
    return torch.stack((boxes_x1, boxes_y1, boxes_x2, boxes_y2), dim=2).reshape(boxes.shape)

def remove_small_boxes(boxes, min_size):
    ws = boxes[:, 2] - boxes[:, 0]
    hs = boxes[:, 3] - boxes[:, 1]
    keep = (ws >= min_size) & (hs >= min_size)
    keep = keep.nonzero().squeeze(1)
    return keep
