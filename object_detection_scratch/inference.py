import torch
from object_detection_scratch.detector import ObjectDetector
from object_detection_scratch.synthetic_dataset import SyntheticDataset
from PIL import Image, ImageDraw
import os
import numpy as np

def draw_boxes(img, boxes, labels, scores, threshold=0.5):
    draw = ImageDraw.Draw(img)
    for i, box in enumerate(boxes):
        if scores[i] < threshold:
            continue
        
        box = box.tolist()
        label = labels[i].item()
        
        # Color based on label
        color = "red" if label == 1 else "blue" # 1=rect, 2=circle
        
        draw.rectangle(box, outline=color, width=3)
        draw.text((box[0], box[1]), f"{scores[i]:.2f}", fill="white")
    return img

def main():
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    
    # Generate one sample
    ds = SyntheticDataset(num_samples=1, transforms=None)
    img, target = ds[0] # PIL image, target dict
    
    # Transform for model
    # Need to mimic validation transform (Resize + ToTensor)
    from object_detection_scratch.dataset import Resize, ToTensor
    # Manually
    img_rs = img.resize((600, 600))
    to_tensor = ToTensor()
    
    # Target boxes need resize too? 
    # Yes, but for inference we just need image.
    
    img_tensor = torch.from_numpy(np.array(img_rs)).permute(2, 0, 1).float() / 255.0
    img_tensor = img_tensor.unsqueeze(0).to(device)
    
    model = ObjectDetector(num_classes=3)
        # Load weights if available
    if os.path.exists("model_epoch_0.pth"):
        model.load_state_dict(torch.load("model_epoch_0.pth", map_location=device))
        print("Loaded model_epoch_0.pth")
    else:
        print("No weights found, using random model.")
        
    model.to(device)
    model.eval()
    
    with torch.no_grad():
        output = model(img_tensor)
        
    # Output is list of dicts
    pred = output[0]
    boxes = pred['boxes'].cpu()
    scores = pred['scores'].cpu()
    labels = pred['labels'].cpu()
    
    print(f"Detections: {len(boxes)}")
    
    # Draw on original image (resized)
    
    res_img = draw_boxes(img_rs, boxes, labels, scores, threshold=0.1)
    res_img.save("result_inference.jpg")
    print("Saved result_inference.jpg")

if __name__ == "__main__":
    import numpy as np
    main()
