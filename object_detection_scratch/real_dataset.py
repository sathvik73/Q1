import torch
from torch.utils.data import Dataset
import numpy as np
from PIL import Image
import skimage.data
import skimage.transform
import random

class RealDataset(Dataset):
    """
    Generates training images using real object templates (Cat, Astronaut)
    placed on random noise/texture backgrounds.
    Classes: 1=Cat, 2=Astronaut
    """
    def __init__(self, size=(600, 600), num_samples=100, transforms=None):
        self.size = size
        self.num_samples = num_samples
        self.transforms = transforms
        
        # Load templates
        # Chelsea is ~300x451
        self.cat_img = Image.fromarray(skimage.data.chelsea())
        # Astronaut is 512x512
        self.astro_img = Image.fromarray(skimage.data.astronaut())
        
        self.templates = {
            1: self.cat_img,
            2: self.astro_img
        }

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Seed
        np.random.seed(idx)
        random.seed(idx)
        
        # Generate background (random noise or simple texture)
        # Using scikit-image data as background would be cool but maybe too easy?
        # Let's use simple noise to make the "real" objects stand out.
        bg_array = np.random.randint(0, 100, (self.size[1], self.size[0], 3), dtype=np.uint8)
        img = Image.fromarray(bg_array, 'RGB')
        
        boxes = []
        labels = []
        
        num_objs = random.randint(1, 3)
        
        for _ in range(num_objs):
            cls_id = random.randint(1, 2)
            template = self.templates[cls_id]
            
            # Random scaling
            scale = random.uniform(0.3, 0.7)
            new_w = int(template.width * scale)
            new_h = int(template.height * scale)
            obj_img = template.resize((new_w, new_h))
            
            # Random Position
            if self.size[0] - new_w > 0:
                x = random.randint(0, self.size[0] - new_w)
            else:
                x = 0
            if self.size[1] - new_h > 0:
                y = random.randint(0, self.size[1] - new_h)
            else:
                y = 0
                
            # Paste
            # Create mask for irregular shapes? No, just rect paste for now.
            # Real images have rectangular borders unless segmented. 
            # For this demo, rectangular paste is fine, or we can use alpha if available? 
            # These images don't have alpha. Rectangular cut and paste.
            img.paste(obj_img, (x, y))
            
            boxes.append([x, y, x+new_w, y+new_h])
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
