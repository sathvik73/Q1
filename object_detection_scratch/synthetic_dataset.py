import torch
from torch.utils.data import Dataset
import numpy as np
from PIL import Image, ImageDraw

class SyntheticDataset(Dataset):
    """
    Generates synthetic images with random shapes (Rectangle, Circle).
    Classes: 1=Rectangle, 2=Circle.
    """
    def __init__(self, size=(600, 600), num_samples=100, transforms=None):
        self.size = size
        self.num_samples = num_samples
        self.transforms = transforms
        self.keep_classes = ['rectangle', 'circle']

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Generate random image
        # Seed for reproducibility based on idx
        np.random.seed(idx)
        
        # Black background
        img = Image.new('RGB', self.size, color=(0, 0, 0))
        draw = ImageDraw.Draw(img)
        
        boxes = []
        labels = []
        
        # Number of objects
        num_objs = np.random.randint(1, 4)
        
        for _ in range(num_objs):
            cls_id = np.random.randint(1, 3) # 1 or 2
            
            # Random position
            w = np.random.randint(50, 150)
            h = np.random.randint(50, 150)
            x1 = np.random.randint(0, self.size[0] - w)
            y1 = np.random.randint(0, self.size[1] - h)
            x2 = x1 + w
            y2 = y1 + h
            
            color = (np.random.randint(50, 255), np.random.randint(50, 255), np.random.randint(50, 255))
            
            if cls_id == 1: # Rectangle
                draw.rectangle([x1, y1, x2, y2], fill=color)
            else: # Circle
                # Fit circle in bounding box
                draw.ellipse([x1, y1, x2, y2], fill=color)
                
            boxes.append([x1, y1, x2, y2])
            labels.append(cls_id)
            
        boxes = torch.tensor(boxes, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.int64)
        
        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = torch.tensor([idx])
        
        if self.transforms:
            img, target = self.transforms(img, target)
            
        return img, target
