import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.ops as ops
from object_detection_scratch.utils import BoxCoder, Matcher, box_iou, nms

class CustomBackbone(nn.Module):
    """
    Simple 5-layer CNN backbone.
    Output features usually downsampled by a factor (e.g. 16 or 32).
    """
    def __init__(self):
        super().__init__()
        # Input: [B, 3, H, W]
        # Layer 1
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        # Layer 2
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        # Layer 3
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        # Layer 4
        self.conv4 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn4 = nn.BatchNorm2d(128)
        # Layer 5
        self.conv5 = nn.Conv2d(128, 256, 3, padding=1)
        self.bn5 = nn.BatchNorm2d(256)
        
        self.pool = nn.MaxPool2d(2, 2)
        self.out_channels = 256

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x)))) # /2
        x = self.pool(F.relu(self.bn2(self.conv2(x)))) # /4
        x = self.pool(F.relu(self.bn3(self.conv3(x)))) # /8
        x = self.pool(F.relu(self.bn4(self.conv4(x)))) # /16
        x = self.pool(F.relu(self.bn5(self.conv5(x)))) # /32
        return x

class RegionProposalNetwork(nn.Module):
    def __init__(self, in_channels, anchor_generator, rpn_head, 
                 fg_iou_thresh=0.7, bg_iou_thresh=0.3,
                 batch_size_per_image=256, positive_fraction=0.5,
                 pre_nms_top_n=2000, post_nms_top_n=1000, nms_thresh=0.7):
        super().__init__()
        self.anchor_generator = anchor_generator
        self.head = rpn_head
        self.box_coder = BoxCoder()
        
        # Hyperparameters
        self.min_size = 1e-3
        self.proposal_matcher = Matcher(
            high_threshold=fg_iou_thresh,
            low_threshold=bg_iou_thresh,
            allow_low_quality_matches=True
        )
        self.fg_bg_sampler = None # Can implement basic balancing in code or separate class
        
        self.batch_size_per_image = batch_size_per_image
        self.positive_fraction = positive_fraction
        
        self.pre_nms_top_n = pre_nms_top_n
        self.post_nms_top_n = post_nms_top_n
        self.nms_thresh = nms_thresh

    def forward(self, images, features, targets=None):
        """
        Args:
            images (ImageList or Tensor): images for which we want to compute the predictions
            features (Tensor): features computed from the images
            targets (List[Dict]): ground-truth boxes present in the image (optional)
        """
        # 1. Generate anchors
        # features is [B, C, H_feat, W_feat]
        # We need to map anchors to the feature map grid.
        feature_map_size = features.shape[-2:]
        image_size = images.shape[-2:] # Assuming standard tensor
        
        anchors = self.anchor_generator(images, features) # [B, N_anchors, 4]
        
        # 2. RPN Head predictions
        objectness, pred_bbox_deltas = self.head(features) 
        # objectness: [B, A, H, W] -> flatten -> [B, N_anchors, 1]
        # pred_bbox_deltas: [B, A*4, H, W] -> flatten -> [B, N_anchors, 4]
        
        objectness = objectness.permute(0, 2, 3, 1).flatten(1) # [B, N_anchors] (assuming 1 logit per anchor)
        # Actually usually 1 value if binary cross entropy (sigmoid), or 2 if softmax.
        # Let's assume sigmoid output (1 channel per anchor) for simplicity.
        
        pred_bbox_deltas = pred_bbox_deltas.permute(0, 2, 3, 1).reshape(objectness.shape[0], -1, 4)
        
        proposals = self.box_coder.decode(pred_bbox_deltas.detach(), anchors)
        proposals = proposals.view(objectness.shape[0], -1, 4)
        
        # 3. Filter proposals (NMS, topk)
        final_proposals = []
        for i in range(images.shape[0]):
            box = proposals[i]
            score = objectness[i].sigmoid() if hasattr(objectness, 'sigmoid') else torch.sigmoid(objectness[i])
            # Clip
            # box = clip_boxes_to_image(box, image_size) 
            # (Assuming clip utility handles single box tensor or batch)
            # Simplified clip:
            h, w = image_size
            box[:, 0].clamp_(min=0, max=w)
            box[:, 1].clamp_(min=0, max=h)
            box[:, 2].clamp_(min=0, max=w)
            box[:, 3].clamp_(min=0, max=h)
            
            # Remove small
            keep = (box[:, 2] - box[:, 0] > self.min_size) & (box[:, 3] - box[:, 1] > self.min_size)
            box = box[keep]
            score = score[keep]
            
            # Pre-NMS top N
            if score.numel() > self.pre_nms_top_n:
                _, keep_top = score.sort(descending=True)
                keep_top = keep_top[:self.pre_nms_top_n]
                box = box[keep_top]
                score = score[keep_top]
                
            # NMS
            keep_nms = nms(box, score, self.nms_thresh)
            keep_nms = keep_nms[:self.post_nms_top_n]
            final_proposals.append(box[keep_nms])
            
        losses = {}
        if self.training:
            assert targets is not None
            labels, matched_gt_boxes = self.assign_targets_to_anchors(anchors, targets)
            regression_targets = self.box_coder.encode(matched_gt_boxes, anchors)
            
            loss_objectness, loss_rpn_box_reg = self.compute_loss(
                objectness, pred_bbox_deltas, labels, regression_targets
            )
            losses = {
                "loss_objectness": loss_objectness,
                "loss_rpn_box_reg": loss_rpn_box_reg
            }
            
        return final_proposals, losses

    def assign_targets_to_anchors(self, anchors, targets):
        labels = []
        matched_gt_boxes = []
        for i in range(len(targets)):
            gt_boxes = targets[i]["boxes"]
            if gt_boxes.numel() == 0:
                # Handle no GT case
                device = anchors.device
                labels.append(torch.zeros(anchors.shape[1], dtype=torch.float32, device=device))
                matched_gt_boxes.append(torch.zeros_like(anchors[i])) # Dummy
                continue

            iou_matrix = box_iou(gt_boxes, anchors[i]) # [N_gt, N_anchors]
            matched_idxs = self.proposal_matcher(iou_matrix) # [N_anchors]
            
            # Construct labels: 1=fg, 0=bg, -1/ -2=ignore
            # Matcher returns index of GT or -1 or -2
            
            # In own simpler matcher:
            # -1: negative
            # -2: ignore
            # >=0: matched index
            
            # Convert to 0/1 targets for objectness
            # We want: 1 for fg, 0 for bg, -1 for ignore
            
            # Where it is >= 0, it is FG
            fg_mask = matched_idxs >= 0
            # Where it is -1, it is BG
            bg_mask = matched_idxs == -1
            
            # Final labels tensor: 1 (fg), 0 (bg), -1 (ignore)
            final_labels = torch.full_like(matched_idxs, -1, dtype=torch.float32)
            final_labels[fg_mask] = 1.0
            final_labels[bg_mask] = 0.0
            
            labels.append(final_labels)
            
            # Matched GT boxes
            # For every anchor, if it's matched to a GT (>=0), we take that GT.
            # If it's BG or ignore, the regression target doesn't matter (masked out in loss)
            # But we need a valid tensor structure.
            clamped_idxs = matched_idxs.clamp(min=0)
            matched_gt = gt_boxes[clamped_idxs] 
            matched_gt_boxes.append(matched_gt)
            
        return torch.stack(labels), torch.stack(matched_gt_boxes)

    def compute_loss(self, objectness, pred_bbox_deltas, labels, regression_targets):
        # inputs: [B, N], [B, N, 4], [B, N], [B, N, 4]
        
        # Sample positive and negative anchors for loss computation
        # (Usually we sub-sample to 256 anchors per image to balance loss)
        
        sampled_pos_inds, sampled_neg_inds = self.sample_pos_neg(labels)
        sampled_inds = sampled_pos_inds | sampled_neg_inds
        
        # Objectness loss (Binary Cross Entropy)
        # Only on sampled indices
        # objectness is logits
        
        # Flatten
        objectness = objectness.flatten()
        labels = labels.flatten()
        sampled_inds = sampled_inds.flatten()
        
        loss_objectness = F.binary_cross_entropy_with_logits(
            objectness[sampled_inds], 
            labels[sampled_inds]
        )
        
        # Regression loss (Smooth L1)
        # Only on positive samples
        pred_bbox_deltas = pred_bbox_deltas.reshape(-1, 4)
        regression_targets = regression_targets.reshape(-1, 4)
        
        pos_inds = sampled_pos_inds.flatten()
        
        if pos_inds.sum() > 0:
            loss_rpn_box_reg = F.smooth_l1_loss(
                pred_bbox_deltas[pos_inds],
                regression_targets[pos_inds],
                beta=1.0 / 9,
                reduction='sum'
            ) / (sampled_inds.sum().float().item() + 1e-5) # Normalize by batch size
        else:
            loss_rpn_box_reg = objectness.sum() * 0 # Dummy zero
            
        return loss_objectness, loss_rpn_box_reg

    def sample_pos_neg(self, labels):
        # labels: [B, N] with 1, 0, -1
        # Implement simplified random sampling
        pos_mask = (labels == 1)
        neg_mask = (labels == 0)
        
        sampled_pos_mask = torch.zeros_like(labels, dtype=torch.bool)
        sampled_neg_mask = torch.zeros_like(labels, dtype=torch.bool)
        
        for i in range(labels.shape[0]):
            pos_idcs = torch.where(pos_mask[i])[0]
            neg_idcs = torch.where(neg_mask[i])[0]
            
            num_pos = int(self.batch_size_per_image * self.positive_fraction)
            num_pos = min(num_pos, len(pos_idcs))
            num_neg = self.batch_size_per_image - num_pos
            num_neg = min(num_neg, len(neg_idcs))
            
            if num_pos > 0:
                perm_pos = torch.randperm(len(pos_idcs), device=labels.device)[:num_pos]
                sampled_pos_mask[i, pos_idcs[perm_pos]] = True
                
            if num_neg > 0:
                perm_neg = torch.randperm(len(neg_idcs), device=labels.device)[:num_neg]
                sampled_neg_mask[i, neg_idcs[perm_neg]] = True
                
        return sampled_pos_mask, sampled_neg_mask

class AnchorGenerator(nn.Module):
    def __init__(self, sizes=((32, 64, 128, 256, 512),), aspect_ratios=((0.5, 1.0, 2.0),)):
        super().__init__()
        self.sizes = sizes # Tuple of tuples
        self.aspect_ratios = aspect_ratios # Tuple of tuples
        
        # Simple implementation for single feature map
        self.cell_anchors = self._generate_anchors_buffers()

    def _generate_anchors_buffers(self):
        # Generate base anchors for one position (0,0)
        # sizes[0] is for the first feature map
        # ...
        anchors = []
        # assume single feature map for now (CustomBackbone returns 1)
        s = self.sizes[0]
        r = self.aspect_ratios[0]
        
        for size in s:
            for ratio in r:
                # w * h = size^2
                # w / h = ratio => w = h * ratio => h * ratio * h = size^2 => h^2 = size^2 / ratio
                h = size /  (ratio ** 0.5)
                w = h * ratio
                
                anchors.append([-w/2, -h/2, w/2, h/2])
        return torch.tensor(anchors, dtype=torch.float32)

    def forward(self, images, features):
        grid_size = features.shape[-2:] # H, W
        image_size = images.shape[-2:]
        stride = [image_size[0] / grid_size[0], image_size[1] / grid_size[1]]
        
        dtype, device = features.dtype, features.device
        
        # Generate grid shifts
        grid_h, grid_w = grid_size
        shift_x = torch.arange(0, grid_w, dtype=dtype, device=device) * stride[1]
        shift_y = torch.arange(0, grid_h, dtype=dtype, device=device) * stride[0]
        
        # Center of pixels
        shift_x = shift_x + stride[1] / 2
        shift_y = shift_y + stride[0] / 2
        
        shift_y, shift_x = torch.meshgrid(shift_y, shift_x, indexing='ij')
        shifts = torch.stack((shift_x, shift_y, shift_x, shift_y), dim=-1).reshape(-1, 4) # [H*W, 4]
        
        # Add base anchors
        cell_anchors = self.cell_anchors.to(device) # [A, 4]
        
        # Broadcast: [H*W, 1, 4] + [1, A, 4] -> [H*W, A, 4] -> [H*W*A, 4]
        all_anchors = (shifts[:, None, :] + cell_anchors[None, :, :]).reshape(-1, 4)
        
        # Expand for batch
        # [B, H*W*A, 4]
        batch_size = features.shape[0]
        return all_anchors.unsqueeze(0).expand(batch_size, -1, -1)

class RPNHead(nn.Module):
    def __init__(self, in_channels, num_anchors):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.cls_logits = nn.Conv2d(in_channels, num_anchors, kernel_size=1)
        self.bbox_pred = nn.Conv2d(in_channels, num_anchors * 4, kernel_size=1)

    def forward(self, x):
        t = F.relu(self.conv(x))
        return self.cls_logits(t), self.bbox_pred(t)

class RoIHead(nn.Module):
    def __init__(self, in_channels, resolution, num_classes):
        super().__init__()
        self.resolution = resolution
        # RoI Align will produce [B*N, C, resolution, resolution]
        
        self.fc6 = nn.Linear(in_channels * resolution * resolution, 1024)
        self.fc7 = nn.Linear(1024, 1024)
        
        self.cls_score = nn.Linear(1024, num_classes)
        self.bbox_pred = nn.Linear(1024, num_classes * 4) 
        # (Standard Faster RCNN has 4 coords per class, or just 4 class agnostic? 
        # Usually 4 * num_classes or 4. Let's do 4 * num_classes for specifics).
        
        self.box_coder = BoxCoder()
        self.proposal_matcher = Matcher(high_threshold=0.5, low_threshold=0.5)
        # 0.5 is standard for RCNN
        self.fg_bg_sampler = None 
        # Implement sampling logic inside forward
        self.score_thresh = 0.05
        self.nms_thresh = 0.5
        self.detections_per_img = 100

    def forward(self, features, proposals, image_shapes, targets=None):
        """
        features: [B, C, H, W]
        proposals: List[Tensor[N, 4]]
        targets: List[Dict]
        """
        
        if self.training:
            assert targets is not None
            proposals, labels, matched_gt_boxes, regression_targets = self.select_training_samples(proposals, targets)
        
        # roi_align
        # proposals is a list of tensors (batch item k has N_k boxes).
        # roi_align expects a single tensor with batch index, or list of tensors.
        # torchvision.ops.roi_align handles list of tensors directly!
        
        # spatial_scale: features H / image H.
        # Assuming features are 1/32 of image
        spatial_scale = 1.0 / 32.0 
        
        box_features = ops.roi_align(
            features, proposals, output_size=(self.resolution, self.resolution),
            spatial_scale=spatial_scale, sampling_ratio=2
        )
        
        # Flatten
        box_features = box_features.flatten(start_dim=1)
        
        # FC
        x = F.relu(self.fc6(box_features))
        x = F.relu(self.fc7(x))
        
        class_logits = self.cls_score(x)
        box_regression = self.bbox_pred(x)
        
        result = []
        losses = {}
        
        if self.training:
            loss_classifier, loss_box_reg = self.compute_loss(class_logits, box_regression, labels, regression_targets)
            losses = {
                "loss_classifier": loss_classifier,
                "loss_box_reg": loss_box_reg
            }
        else:
            boxes, scores, labels = self.postprocess_detections(class_logits, box_regression, proposals, image_shapes)
            num_images = len(image_shapes)
            for i in range(num_images):
                result.append(
                    {
                        "boxes": boxes[i],
                        "labels": labels[i],
                        "scores": scores[i],
                    }
                )

        return result, losses

    def select_training_samples(self, proposals, targets):
        # Sample FG/BG for RCNN head
        # Usually 512 ROIs per image, 25% FG
        
        final_proposals = []
        final_labels = []
        final_regression_targets = []
        final_matched_gt_boxes = []
        
        batch_size_per_image = 512
        positive_fraction = 0.25
        
        for i in range(len(targets)):
            gt_boxes = targets[i]["boxes"]
            gt_labels = targets[i]["labels"] # 0 is background? NO. Usually 1...K are classes. 0 is BG.
            # But in dataset.py I used 1-based indexing for subset.
            
            # Start by adding GT boxes to proposals to ensure some positives
            props = torch.cat([proposals[i], gt_boxes], dim=0)

            iou_matrix = box_iou(gt_boxes, props)
            matched_idxs = self.proposal_matcher(iou_matrix) # [N_props]
            
            # Label assignment
            # indices >= 0 are matched to GT index
            # -1 is negative (IoU < 0.5)
            # -2 is ignore? Matcher implementation might have specific behavior.
            # RCNN typically: IoU >= 0.5 -> FG. [0.1, 0.5) -> BG. < 0.1 -> Ignore? Or just all BG.
            # Simplified: >= 0.5 FG, else BG.
            
            # Matcher implementation:
            # matched < low (0.5) -> -1
            # matched >= high (0.5) -> index
            # So binary here.
            
            # Prepare targets
            # FG labels = gt_labels[matched_idxs]
            # BG labels = 0
            
            # Handle -1 (BG)
            bg_inds = (matched_idxs == -1)
            fg_inds = (matched_idxs >= 0)
            
            # Sample
            pos_idcs = torch.where(fg_inds)[0]
            neg_idcs = torch.where(bg_inds)[0]
            
            num_pos = int(batch_size_per_image * positive_fraction)
            num_pos = min(num_pos, len(pos_idcs))
            num_neg = batch_size_per_image - num_pos
            num_neg = min(num_neg, len(neg_idcs))
            
            sampled_pos_inds = pos_idcs[torch.randperm(len(pos_idcs))[:num_pos]]
            sampled_neg_inds = neg_idcs[torch.randperm(len(neg_idcs))[:num_neg]]
            
            sampled_inds = torch.cat([sampled_pos_inds, sampled_neg_inds])
            
            # Gather
            sampled_props = props[sampled_inds]
            matched_idxs_sampled = matched_idxs[sampled_inds]
            
            sampled_labels = torch.zeros(len(sampled_inds), dtype=torch.int64, device=props.device) # Default 0 (BG)
            
            # Fill FG labels
            # matched_idxs_sampled >= 0 for FGs
            fg_in_sample = matched_idxs_sampled >= 0
            if fg_in_sample.any():
                matched_gt_inds = matched_idxs_sampled[fg_in_sample]
                sampled_labels[fg_in_sample] = gt_labels[matched_gt_inds]
            
            # Regression targets
            # For BGs, regression target is dummy
            sampled_gt_boxes = torch.zeros_like(sampled_props)
            if fg_in_sample.any():
                matched_gt_inds = matched_idxs_sampled[fg_in_sample]
                sampled_gt_boxes[fg_in_sample] = gt_boxes[matched_gt_inds]
                
            sampled_reg_targets = self.box_coder.encode(sampled_gt_boxes, sampled_props)
            
            final_proposals.append(sampled_props)
            final_labels.append(sampled_labels)
            final_regression_targets.append(sampled_reg_targets)
            final_matched_gt_boxes.append(sampled_gt_boxes) # Not really needed except for verifying
        
        return final_proposals, torch.cat(final_labels), final_matched_gt_boxes, torch.cat(final_regression_targets)

    def compute_loss(self, class_logits, box_regression, labels, regression_targets):
        # class_logits: [Sum(N), NumClasses]
        # labels: [Sum(N)]
        
        loss_classifier = F.cross_entropy(class_logits, labels)
        
        # Split box_regression: [Sum(N), NumClasses*4]
        # We only want loss for the specific class gt
        
        # Get FG samples only for regression
        fg_inds = (labels > 0)
        
        if fg_inds.sum() == 0:
            return loss_classifier, class_logits.sum() * 0
        
        fg_logits = box_regression[fg_inds]
        fg_labels = labels[fg_inds]
        fg_targets = regression_targets[fg_inds]
        
        # Reshape to [N_fg, NumClasses, 4]
        fg_logits = fg_logits.reshape(fg_inds.sum(), -1, 4)
        
        # Select the specific class logits
        # Gather logic: gather along dim 1
        # fg_labels is [N_fg]
        # We want [N_fg, 1, 4] -> indices [N_fg, 1, 4] essentially? No.
        
        # Just loop or improved indexing
        # fg_logits[i, fg_labels[i], :]
        
        # Easy way:
        indices = torch.arange(fg_inds.sum(), device=fg_logits.device)
        pred_boxes = fg_logits[indices, fg_labels] # [N_fg, 4]
        
        loss_box_reg = F.smooth_l1_loss(pred_boxes, fg_targets, beta=1.0, reduction='sum') / labels.numel()
        
        return loss_classifier, loss_box_reg

    def postprocess_detections(self, class_logits, box_regression, proposals, image_shapes):
        # class_logits: [TotalN, NumClasses]
        # proposals: List[Tensor] (need to unpack)
        
        device = class_logits.device
        num_classes = class_logits.shape[-1]
        
        probs = F.softmax(class_logits, dim=-1)
        
        # box_regression: [TotalN, NumClasses*4]
        box_regression = box_regression.reshape(box_regression.shape[0], -1, 4)
        
        # We have proposals concatenated across batch. We need to split back.
        boxes_per_image = [p.shape[0] for p in proposals]
        probs_split = probs.split(boxes_per_image, dim=0)
        box_reg_split = box_regression.split(boxes_per_image, dim=0)
        
        all_boxes = []
        all_scores = []
        all_labels = []
        
        for i, (prob, box_deltas, props) in enumerate(zip(probs_split, box_reg_split, proposals)):
            # prob: [N, K], box_deltas: [N, K, 4], props: [N, 4]
            # Apply deltas to proposals
            # We need to expand props to [N, K, 4]
            
            # Apply decoder
            # To vectorize: flatten N, K
            
            # For each class k > 0 (background is 0)
            
            boxes_img = []
            scores_img = []
            labels_img = []
            
            for k in range(1, num_classes):
                # Get scores for class k
                scores_k = prob[:, k]
                
                # Filter low scores early
                keep_idxs = scores_k > self.score_thresh
                if not keep_idxs.any():
                    continue
                    
                scores_k = scores_k[keep_idxs]
                deltas_k = box_deltas[keep_idxs, k]
                props_k = props[keep_idxs]
                
                decoded_boxes = self.box_coder.decode(deltas_k, props_k)
                
                # Clip
                h, w = image_shapes[i]
                decoded_boxes[:, 0].clamp_(min=0, max=w)
                decoded_boxes[:, 1].clamp_(min=0, max=h)
                decoded_boxes[:, 2].clamp_(min=0, max=w)
                decoded_boxes[:, 3].clamp_(min=0, max=h)
                
                # NMS
                keep_nms = nms(decoded_boxes, scores_k, self.nms_thresh)
                keep_nms = keep_nms[:self.detections_per_img]
                
                boxes_img.append(decoded_boxes[keep_nms])
                scores_img.append(scores_k[keep_nms])
                labels_img.append(torch.full((len(keep_nms),), k, dtype=torch.int64, device=device))
                
            if len(boxes_img) > 0:
                all_boxes.append(torch.cat(boxes_img))
                all_scores.append(torch.cat(scores_img))
                all_labels.append(torch.cat(labels_img))
            else:
                all_boxes.append(torch.empty((0, 4), device=device))
                all_scores.append(torch.empty((0,), device=device))
                all_labels.append(torch.empty((0,), dtype=torch.int64, device=device))
        
        return all_boxes, all_scores, all_labels
