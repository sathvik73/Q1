import torch
import numpy as np
from PIL import Image, ImageDraw
import skimage.data
from object_detection_scratch.detector import ObjectDetector
from object_detection_scratch.dataset import Resize, ToTensor
import os

def get_model(device):
    model = ObjectDetector(num_classes=3) # 1=Cat, 2=Astro, 3=Dummy(unused)
    if os.path.exists("model_epoch_4.pth"): # Load last epoch
        model.load_state_dict(torch.load("model_epoch_4.pth", map_location=device))
    elif os.path.exists("model_epoch_0.pth"):
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
        
        # 1=Cat (Green), 2=Astronaut (Orange)
        label_map = {1: "Cat", 2: "Astronaut"}
        name = label_map.get(label, "Unknown")
        color = "green" if label == 1 else "orange"
        
        draw.rectangle(box, outline=color, width=3)
        draw.text((box[0], box[1]), f"{name} {score:.2f}", fill="white")
    return img

def make_frame(t, cat_img, astro_img, size=(600, 600)):
    # Simple background
    img = Image.new('RGB', size, color=(50, 50, 50))
    
    # Move Cat
    # Sine wave motion
    scale_cat = 0.5
    c_w, c_h = int(cat_img.width * scale_cat), int(cat_img.height * scale_cat)
    cat_resized = cat_img.resize((c_w, c_h))
    
    x1 = int(100 + t * 5) % (size[0] - c_w)
    y1 = int(300 + 100 * np.sin(t * 0.2))
    img.paste(cat_resized, (x1, y1))
    
    # Move Astronaut
    scale_astro = 0.4
    a_w, a_h = int(astro_img.width * scale_astro), int(astro_img.height * scale_astro)
    astro_resized = astro_img.resize((a_w, a_h))
    
    x2 = int(400 - t * 5) % (size[0] - a_w)
    y2 = int(100 + 50 * np.cos(t * 0.2))
    img.paste(astro_resized, (x2, y2))
    
    return img

def main():
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = get_model(device)
    
    # Load templates
    cat_img = Image.fromarray(skimage.data.chelsea())
    astro_img = Image.fromarray(skimage.data.astronaut())
    
    frames_orig = []
    frames_det = []
    
    print("Generating real frames...")
    # Generate 40 frames
    for t in range(40):
        img_orig = make_frame(t, cat_img, astro_img)
        frames_orig.append(img_orig)
        
        img_tensor = torch.from_numpy(np.array(img_orig)).permute(2, 0, 1).float() / 255.0
        img_tensor = img_tensor.unsqueeze(0).to(device)
        
        with torch.no_grad():
            output = model(img_tensor)
            
        pred = output[0]
        boxes = pred['boxes'].cpu()
        scores = pred['scores'].cpu()
        labels = pred['labels'].cpu()
        
        # Draw
        print(f"Frame {t}: Found {len(boxes)} boxes (Threshold 0.1)")
        img_drawn = draw_boxes(img_orig.copy(), boxes, labels, scores, threshold=0.1)
        frames_det.append(img_drawn)
        
    # Save GIFs
    print("Saving GIFs...")
    frames_orig[0].save(
        "real_original.gif",
        save_all=True,
        append_images=frames_orig[1:],
        optimize=False,
        duration=100,
        loop=0
    )
    print("Saved real_original.gif")
    
    frames_det[0].save(
        "real_detection.gif",
        save_all=True,
        append_images=frames_det[1:],
        optimize=False,
        duration=100,
        loop=0
    )
    print("Saved real_detection.gif")

if __name__ == "__main__":
    main()
