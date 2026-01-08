import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from object_detection_scratch.dataset import get_transforms, collate_fn
from object_detection_scratch.real_dataset import RealDataset
from object_detection_scratch.detector import ObjectDetector
import time
import math

def train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10):
    model.train()
    metric_logger = []
    
    header = 'Epoch: [{}]'.format(epoch)
    
    lr_scheduler = None
    if epoch == 0:
        warmup_factor = 1.0 / 1000
        warmup_iters = min(1000, len(data_loader) - 1)
        
        # Simple warmup
        lr_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=warmup_factor, total_iters=warmup_iters
        )

    for i, (images, targets) in enumerate(data_loader):
        images = images.to(device)
        # targets is list of dicts. Move tensors to device
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        
        # Check shapes
        # If batching fails due to different sizes, we handled resize in dataset?
        # Dataset.py doesn't have Resize transform in "get_transforms" yet!
        # It only has ToTensor. "pass" for train.
        # This will fail batching if images are different sizes and we stack them?
        # WAIT. 
        # collate_fn return tuple(zip(*batch)). So images is a tuple of tensors.
        # detector.forward expects "List[Tensor] or Tensor". 
        # If List[Tensor], it works for variable sizes.
        # BUT, my Backbone (CNN) uses standard Conv2d which preserves spatial dimensions relative to input.
        # RPN/RoI handles variable sizes via spatial_scale?
        # Yes, standard R-CNN handles variable image sizes by processing one by one or padding.
        # My detector.forward:
        # 1. Normalize.
        # images is list of tensors?
        # If so: (images - mean) -> will fail if shapes differ?
        # No, loop over list.
        # Check detector.forward: 
        # "Assuming images are already collated into a Tensor [B, 3, H, W] for simplicity"
        # I wrote that comment. I need to handle List[Tensor].
        # I should fix detector logic or ensure fixed size resize.
        # Fixed size resize is easier for "scratch" implementation optimization context (GPU batching).
        # Let's fix dataset to resize to 600x600 for simplicity.
        
        loss_dict = model(images, targets)
        
        losses = sum(loss for loss in loss_dict.values())
        
        loss_dict_reduced = {k: v.item() for k, v in loss_dict.items()}
        loss_value = losses.item()
        
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()
        
        if lr_scheduler is not None:
            lr_scheduler.step()
            
        metric_logger.append(loss_value)
        
        if i % print_freq == 0:
            print(f"Epoch: [{epoch}] [{i}/{len(data_loader)}] Loss: {loss_value:.4f} "
                  f"({torch.tensor(metric_logger).mean():.4f}) "
                  f"Loc: {loss_dict_reduced['loss_rpn_box_reg'] + loss_dict_reduced['loss_box_reg']:.4f} "
                  f"Cls: {loss_dict_reduced['loss_objectness'] + loss_dict_reduced['loss_classifier']:.4f}")

def main():
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f"Using device: {device}")
    
    # Dataset
    # We need to enforce Resize in dataset transform to support batching as Tensor 
    # OR update detector to stack images with padding.
    # For simplicity, let's update dataset.py to resize to fixed size (e.g. 448x448 or 512x512).
    # But wait, resizing also requires resizing boxes.
    # torchvision.transforms.Resize doesn't resize boxes automatically (unless using v2).
    # I'll rely on batch size = 1 if sizes differ, or implement resize manually.
    # PROPOSAL: Use batch_size=2 and assume similar sizes or custom collate that stacks padded?
    # Torchvision References utils.collate_fn does not stack, it assumes list of images.
    # My detector.forward said: "Assuming images are already collated into a Tensor".
    # I should change that to handle list of tensors (nested tensor approach or loop).
    # Easier: Just Stack them? No, shapes differ.
    # I will stick to batch_size=1 for safety if I don't resize? Slow.
    # Better: Update detector to nested_tensor_from_tensor_list approach (padding).
    # Or just resize.
    
    # Let's try batch_size=2 and see. If it fails, I'll fix.
    # Ideally, I'd update detector.py to handle List[Tensor] by padding them into a batch.
    
    # Use RealDataset
    dataset_train = RealDataset(num_samples=100, transforms=get_transforms(train=True))
    dataset_val = RealDataset(num_samples=20, transforms=get_transforms(train=False))
    
    data_loader = DataLoader(
        dataset_train, batch_size=2, shuffle=True,  # Small batch because of unoptimized padding/memory
        collate_fn=collate_fn, num_workers=0
    )
    
    model = ObjectDetector(num_classes=3)
    model.to(device)
    
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
    
    num_epochs = 5
    
    for epoch in range(num_epochs):
        train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10)
        
        # Save checkopint
        if epoch % 1 == 0:
            torch.save(model.state_dict(), f"model_epoch_{epoch}.pth")
            print(f"Saved model_epoch_{epoch}.pth")

if __name__ == "__main__":
    main()
