import torch
import torchvision
from torchvision.datasets import VOCDetection
import torchvision.transforms.functional as F
import os
from PIL import Image

class VOCSubset(VOCDetection):
    """
    Custom PASCAL VOC dataset that filters for specific classes.
    """
    def __init__(self, root, year='2007', image_set='train', download=False, transforms=None):
        super().__init__(root, year=year, image_set=image_set, download=download, transform=None, target_transform=None)
        
        self.keep_classes = ['person', 'car', 'dog']
        self.class_to_idx = {cls: i+1 for i, cls in enumerate(self.keep_classes)} # 0 is background
        self.transforms = transforms
        
        # Filter images that don't contain any of the keep_classes
        self.filtered_indices = []
        print(f"Filtering dataset for classes: {self.keep_classes}...")
        for i in range(len(self.images)):
            # We need to parse to check existence. 
            # Optimization: could skip this and just return empty in getitem, but len() would be wrong.
            # Using the cached parse if possible or just parse.
            try:
                target = self.parse_voc_xml(self.annotations[i])
                objects = target['annotation']['object']
                if not isinstance(objects, list):
                    objects = [objects]
                
                has_class = False
                for obj in objects:
                    if obj['name'] in self.keep_classes:
                        has_class = True
                        break
                if has_class:
                    self.filtered_indices.append(i)
            except Exception as e:
                # Handle cases where xml might be malformed or different structure
                pass
                
        print(f"Kept {len(self.filtered_indices)} images out of {len(self.images)}")

    def __len__(self):
        return len(self.filtered_indices)

    def __getitem__(self, idx):
        real_idx = self.filtered_indices[idx]
        img = Image.open(self.images[real_idx]).convert('RGB')
        target = self.parse_voc_xml(self.annotations[real_idx])
        
        boxes = []
        labels = []
        
        objects = target['annotation']['object']
        if not isinstance(objects, list):
            objects = [objects]
            
        for obj in objects:
            cls_name = obj['name']
            if cls_name in self.keep_classes:
                bndbox = obj['bndbox']
                bbox = [
                    float(bndbox['xmin']),
                    float(bndbox['ymin']),
                    float(bndbox['xmax']),
                    float(bndbox['ymax'])
                ]
                boxes.append(bbox)
                labels.append(self.class_to_idx[cls_name])
        
        boxes = torch.tensor(boxes, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.int64)
        
        target_dict = {}
        target_dict["boxes"] = boxes
        target_dict["labels"] = labels
        target_dict["image_id"] = torch.tensor([idx])
        
        if self.transforms:
            img, target_dict = self.transforms(img, target_dict)
            
        return img, target_dict

class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target

class ToTensor:
    def __call__(self, image, target):
        image = F.to_tensor(image)
        return image, target

class Resize:
    def __init__(self, size=(600, 600)):
        self.size = size # (h, w)

    def __call__(self, image, target):
        # image is PIL or Tensor
        # Resize image
        old_w, old_h = image.size
        new_h, new_w = self.size
        
        image = F.resize(image, self.size)
        
        # Resize boxes
        boxes = target["boxes"]
        if boxes.numel() > 0:
            scale_x = new_w / old_w
            scale_y = new_h / old_h
            
            # xmin, ymin, xmax, ymax
            boxes[:, 0] *= scale_x
            boxes[:, 1] *= scale_y
            boxes[:, 2] *= scale_x
            boxes[:, 3] *= scale_y
            
        target["boxes"] = boxes
        return image, target

def get_transforms(train):
    transforms = []
    transforms.append(Resize((600, 600)))
    transforms.append(ToTensor())
    return Compose(transforms)

# Simple Collate fn
def collate_fn(batch):
    # Zip images and targets
    images, targets = zip(*batch)
    # Stack images since they are same size now
    images = torch.stack(images, dim=0)
    return images, targets

if __name__ == "__main__":
    # Test
    ds = VOCSubset(root="./data", download=True, transforms=get_transforms(train=True))
    print(f"Dataset length: {len(ds)}")
    if len(ds) > 0:
        img, target = ds[0]
        print(f"Image shape: {img.shape}")
        print(f"Target boxes: {target['boxes']}")
