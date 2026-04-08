import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
from tqdm import tqdm

MODEL_PATH = 'models/resnet50/eurosat_resnet50_final.pth'
PATCHES_DIR = 'data/Lahore_Dataset/patches/'
RESULTS_DIR = 'results/'
OUTPUT_IMAGE = os.path.join(RESULTS_DIR, 'lahore_predictions_sample.png')
EUROSAT_CLASSES = ['AnnualCrop', 'Forest', 'HerbaceousVegetation', 'Highway', 'Industrial', 'Pasture', 'PermanentCrop', 'Residential', 'River', 'SeaLake']

def load_model(path):
    model = models.resnet50(weights=None)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, len(EUROSAT_CLASSES))
    checkpoint = torch.load(path, map_location='cpu')
    if isinstance(checkpoint, dict):
        if 'model_state' in checkpoint: model.load_state_dict(checkpoint['model_state'])
        elif 'state_dict' in checkpoint: model.load_state_dict(checkpoint['state_dict'])
        else: model.load_state_dict(checkpoint)
    else: model = checkpoint
    model.eval()
    return model

def get_transform():
    return transforms.Compose([transforms.Resize((64, 64)), transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

def run_inference():
    if not os.path.exists(RESULTS_DIR): os.makedirs(RESULTS_DIR)
    model = load_model(MODEL_PATH)
    transform = get_transform()
    files = glob.glob(os.path.join(PATCHES_DIR, '*.jpg'))[:100] # Sample 100
    results = []
    with torch.no_grad():
        for file_path in tqdm(files):
            try:
                img = Image.open(file_path).convert('RGB')
                img_t = transform(img).unsqueeze(0)
                output = model(img_t)
                probs = torch.nn.functional.softmax(output, dim=1)
                conf, pred = torch.max(probs, 1)
                results.append({'filename': os.path.basename(file_path), 'class': EUROSAT_CLASSES[pred.item()], 'confidence': conf.item(), 'image': img})
            except Exception as e: print(f"Error {file_path}: {e}")
    return results

def visualize(results):
    print("\n--- Sample Predictions ---")
    class_counts = {cls: 0 for cls in EUROSAT_CLASSES}
    for res in results:
        print(f"{res['filename']} | {res['class']} | {res['confidence']*100:.2f}%")
        class_counts[res['class']] += 1
    print("\n--- Distribution ---")
    for cls, count in class_counts.items(): print(f"{cls}: {count}")
    
    sample_size = min(25, len(results))
    cols = 5
    rows = (sample_size + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(15, 3*rows))
    axes = axes.flatten() if rows * cols > 1 else [axes]
    for i in range(sample_size):
        res = results[i]
        axes[i].imshow(res['image'])
        axes[i].set_title(f"{res['class']}\n{res['confidence']*100:.1f}%", fontsize=10)
        axes[i].axis('off')
    plt.tight_layout()
    plt.savefig(OUTPUT_IMAGE)

if __name__ == "__main__":
    res = run_inference()
    visualize(res)
