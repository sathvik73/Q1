import torch
import numpy as np
from PIL import Image, ImageDraw
from object_detection_scratch.detector import ObjectDetector
from object_detection_scratch.dataset import Resize, ToTensor
import os

def get_model(device):
    model = ObjectDetector(num_classes=3)
    if os.path.exists("model_epoch_0.pth"):
        model.load_state_dict(torch.load("model_epoch_0.pth", map_location=device))
    model.to(device)
    model.eval()
    return model

def draw_boxes(img, boxes, labels, scores, threshold=0.5):
    draw = ImageDraw.Draw(img)
    for i, box in enumerate(boxes):
        if scores[i] < threshold:
            continue
        box = box.tolist()
        label = labels[i].item()
        score = scores[i].item()
        
        # 1=Rect (Red), 2=Circle (Blue)
        color = "red" if label == 1 else "blue"
        
        draw.rectangle(box, outline=color, width=3)
        draw.text((box[0], box[1]), f"{score:.2f}", fill="white")
    return img

def make_frame(t, step, size=(600, 600)):
    # Create a frame where objects move based on t
    img = Image.new('RGB', size, color=(0, 0, 0))
    draw = ImageDraw.Draw(img)
    
    # Object 1: Rectangle moving right
    x1 = (50 + t * 10) % (size[0] - 100)
    y1 = 100
    w, h = 100, 80
    draw.rectangle([x1, y1, x1+w, y1+h], fill=(200, 50, 50))
    
    # Object 2: Circle moving diagonal
    x2 = (400 - t * 8) % (size[0] - 80)
    y2 = (50 + t * 5) % (size[1] - 80)
    w2 = 80
    draw.ellipse([x2, y2, x2+w2, y2+w2], fill=(50, 50, 200))
    
    return img

def main():
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = get_model(device)
    
    frames = []
    resize = Resize((600, 600))
    to_tensor = ToTensor()
    
    print("Generating frames...")
    # Generate 30 frames
    for t in range(30):
        img_orig = make_frame(t, 0)
        
        # Preprocess
        # For simplicity, inference just needs tensor.
        # Resize is identity here since we make 600x600 frames.
        
        img_tensor = torch.from_numpy(np.array(img_orig)).permute(2, 0, 1).float() / 255.0
        img_tensor = img_tensor.unsqueeze(0).to(device)
        
        with torch.no_grad():
            output = model(img_tensor)
            
        pred = output[0]
        boxes = pred['boxes'].cpu()
        scores = pred['scores'].cpu()
        labels = pred['labels'].cpu()
        
        # Draw
        img_drawn = draw_boxes(img_orig.copy(), boxes, labels, scores, threshold=0.3)
        frames.append(img_drawn)
        
    # Save GIF
    print("Saving GIF...")
    frames[0].save(
        "detection_demo.gif",
        save_all=True,
        append_images=frames[1:],
        optimize=False,
        duration=100,
        loop=0
    )
    print("Saved detection_demo.gif")

if __name__ == "__main__":
    main()
