import torch
import torch.nn as nn
from object_detection_scratch.model_components import CustomBackbone, RegionProposalNetwork, AnchorGenerator, RPNHead, RoIHead

class ObjectDetector(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()
        # Backbone
        self.backbone = CustomBackbone()
        out_channels = self.backbone.out_channels
        
        # RPN
        # Anchor sizes for a single feature map suitable for VOC (images ~500x500)
        # stride is 32. 
        # small objects might be lost. 
        # 32*32 = 1024 pixels. 
        # anchors: 128, 256, 512 might be appropriate?
        anchor_sizes = ((64, 128, 256, 512),) 
        aspect_ratios = ((0.5, 1.0, 2.0),)
        
        anchor_generator = AnchorGenerator(sizes=anchor_sizes, aspect_ratios=aspect_ratios)
        num_anchors = len(anchor_sizes[0]) * len(aspect_ratios[0])
        
        rpn_head = RPNHead(out_channels, num_anchors)
        
        self.rpn = RegionProposalNetwork(
            out_channels, anchor_generator, rpn_head,
            fg_iou_thresh=0.7, bg_iou_thresh=0.3,
            batch_size_per_image=256, positive_fraction=0.5,
            pre_nms_top_n=2000, post_nms_top_n=1000, nms_thresh=0.7
        )
        
        # RoI Head
        # Resolution 7 is standard
        # num_classes + 1 (for background) = 4
        # But wait, num_classes passed should be actual classes?
        # In dataset.py I filtered for 3 classes: Person, Car, Dog.
        # Plus background = 4 classes total.
        # My dataset returns 1, 2, 3. 0 is background.
        # So we need 4 output logits.
        
        self.roi_heads = RoIHead(
            out_channels, resolution=7, num_classes=num_classes + 1
        )
        
        # Normalization (optional). Training from scratch, maybe careful.
        # Standard mean/std
        self.image_mean = [0.485, 0.456, 0.406]
        self.image_std = [0.229, 0.224, 0.225]

    def forward(self, images, targets=None):
        """
        images: List[Tensor] or Tensor [B, 3, H, W] (if resized to same size)
        targets: List[Dict]
        """
        # 1. Preprocess (Normalize)
        # Assuming images are already collated into a Tensor [B, 3, H, W] for simplicity
        # (Since we used resize in transforms)
        
        # Normalize
        mean = torch.as_tensor(self.image_mean, dtype=images.dtype, device=images.device)[None, :, None, None]
        std = torch.as_tensor(self.image_std, dtype=images.dtype, device=images.device)[None, :, None, None]
        images = (images - mean) / std
        
        # 2. Backbone
        features = self.backbone(images) # [B, 256, H/32, W/32]
        
        # 3. RPN
        proposals, rpn_losses = self.rpn(images, features, targets)
        
        # 4. RoI Heads
        # Proposals is list of boxes
        # Image shapes needed for post processing (clipping)
        image_shapes = [img.shape[-2:] for img in images]
        
        result, roi_losses = self.roi_heads(features, proposals, image_shapes, targets)
        
        if self.training:
            losses = {}
            losses.update(rpn_losses)
            losses.update(roi_losses)
            return losses
        else:
            return result
