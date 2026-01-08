import torch
from object_detection_scratch.detector import ObjectDetector
import os
from object_detection_scratch.dataset import get_transforms, collate_fn
from object_detection_scratch.synthetic_dataset import SyntheticDataset
from torch.utils.data import DataLoader
import time
from collections import defaultdict
import numpy as np

def calculate_map(detections, ground_truths, iou_threshold=0.5):
    """
    Simple mAP calculation for 3 classes.
    detections: List of dicts per image.
    ground_truths: List of dicts per image.
    """
    # ... Simplified mAP implementation ...
    # This is non-trivial to implement from scratch correctly (11-point or all-point interpolation).
    # For this task, we can use torchmetrics if available or a simple approximation.
    # Since we are "scratch", I'll write a simple one.
    
    average_precisions = {}
    
    # Class-wise
    # 1=Person, 2=Car, 3=Dog (assuming 1-based from dataset)
    # But dataset returns 1, 2, 3.
    
    for class_id in [1, 2, 3]:
        true_positives = []
        scores = []
        num_gt = 0
        
        for i in range(len(detections)):
            # Get dets and gts for this image and class
            pred_boxes = []
            pred_scores = []
            
            det = detections[i]
            # Filter class
            mask = det['labels'] == class_id
            if mask.any():
                pred_boxes = det['boxes'][mask]
                pred_scores = det['scores'][mask]
                
            gt = ground_truths[i]
            gt_mask = gt['labels'] == class_id
            gt_boxes = gt['boxes'][gt_mask]
            
            num_gt += len(gt_boxes)
            
            # Sort preds by score
            if len(pred_boxes) == 0:
                continue
                
            sorted_idxs = torch.argsort(pred_scores, descending=True)
            pred_boxes = pred_boxes[sorted_idxs]
            pred_scores = pred_scores[sorted_idxs]
            
            used_gt = torch.zeros(len(gt_boxes), dtype=torch.bool)
            
            for b_idx, box in enumerate(pred_boxes):
                scores.append(pred_scores[b_idx].item())
                
                if len(gt_boxes) == 0:
                    true_positives.append(0)
                    continue
                
                # IoU
                ious = torch.ops.torchvision.box_iou(box.unsqueeze(0), gt_boxes)
                max_iou, max_idx = ious.max(dim=1)
                
                if max_iou > iou_threshold:
                    if not used_gt[max_idx]:
                        true_positives.append(1)
                        used_gt[max_idx] = True
                    else:
                        true_positives.append(0) # Duplicate detection
                else:
                    true_positives.append(0)
                    
        if num_gt == 0:
            continue
            
        # Compute AP
        scores = np.array(scores)
        true_positives = np.array(true_positives)
        
        # Sort all by score
        indices = np.argsort(-scores)
        true_positives = true_positives[indices]
        
        accumulated_tp = np.cumsum(true_positives)
        accumulated_fp = np.cumsum(1 - true_positives)
        
        precision = accumulated_tp / (accumulated_tp + accumulated_fp + 1e-6)
        recall = accumulated_tp / (num_gt + 1e-6)
        
        # 11-point interpolation or AUC
        ap = np.trapz(precision, recall) # Simple AUC
        average_precisions[class_id] = ap
        
    mAP = np.mean(list(average_precisions.values())) if average_precisions else 0.0
    return mAP

def evaluate(model, data_loader, device):
    model.eval()
    detections = []
    ground_truths = []
    
    inference_times = []
    
    with torch.no_grad():
        for i, (images, targets) in enumerate(data_loader):
            if i >= 50: break # Limit evaluation for speed
            
            images = images.to(device)
            # targets already list of dicts mostly
            
            start_time = time.time()
            outputs = model(images)
            end_time = time.time()
            if i > 0: # warm up
                inference_times.append(end_time - start_time)
            
            # Unpack
            # outputs is list of dicts
            for j in range(len(outputs)):
                detections.append({k: v.cpu() for k, v in outputs[j].items()})
                ground_truths.append({k: v.cpu() for k, v in targets[j].items()})
                
    fps = 1.0 / np.mean(inference_times) if inference_times else 0.0
    
    map_score = calculate_map(detections, ground_truths)
    
    print(f"mAP: {map_score:.4f}")
    print(f"FPS: {fps:.2f}")
    return map_score, fps

def main():
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f"Evaluating on {device}")
    
    dataset_val = SyntheticDataset(num_samples=50, transforms=get_transforms(train=False))
    data_loader = DataLoader(dataset_val, batch_size=1, shuffle=False, collate_fn=collate_fn)
    
    model = ObjectDetector(num_classes=3)
    # Load weights if available
    if os.path.exists("model_epoch_0.pth"):
        model.load_state_dict(torch.load("model_epoch_0.pth", map_location=device))
        print("Loaded model_epoch_0.pth")
    else:
        print("No weights found, evaluating random model.")
        
    model.to(device)
    
    evaluate(model, data_loader, device)
    
    # Model size
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    size_all_mb = (param_size + buffer_size) / 1024**2
    print(f'Model size: {size_all_mb:.2f} MB')

if __name__ == "__main__":
    main()
