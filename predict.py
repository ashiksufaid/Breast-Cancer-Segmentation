import os
import argparse
from PIL import Image
import torch
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from model import UNet
from config import device

device = device
model = UNet(3, 1)
checkpoint_path = os.path.join(os.getcwd(), 'checkpoints', 'unet_breast_cancer_segmentation.pth')
state_dict = torch.load(checkpoint_path, map_location=device)
model.load_state_dict(state_dict)
model = model.to(device)
model.eval()

folder = os.path.join(os.getcwd(), "data")
def plot_segmented_images(folder=folder, max_plots=5):
    """
    Scans `folder` for image files, runs UNet segmentation, and plots results.
    """
    # Collect image file paths
    image_paths = []
    for fname in os.listdir(folder):
        lower = fname.lower()
        if not lower.endswith(('.png', '.jpg', '.jpeg')):
            continue
        if '_mask' in lower:
            continue
        full_path = os.path.join(folder, fname)
        if os.path.isfile(full_path):
            image_paths.append(full_path)
    if not image_paths:
        print(f"No valid image files found in '{folder}'.")
        return

    image_paths = image_paths[:max_plots]
    print(f"Processing {len(image_paths)} images from '{folder}'")

    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    images = []
    for img_path in image_paths:
        img = Image.open(img_path).convert('RGB')
        images.append(transform(img))

    batch = torch.stack(images).to(device)

    with torch.no_grad():
        outputs = model(batch)
        outputs = torch.sigmoid(outputs)
        preds = (outputs > 0.5).float().cpu()

    imgs_cpu = (batch.cpu() * 0.5 + 0.5)

    fig, axes = plt.subplots(len(imgs_cpu), 2, figsize=(8, 4 * len(imgs_cpu)))
    if len(imgs_cpu) == 1:
        axes = [axes]

    for idx, (inp, pred) in enumerate(zip(imgs_cpu, preds)):
        img_np = inp.permute(1, 2, 0).numpy()
        mask_np = pred.squeeze().numpy()

        axes[idx][0].imshow(img_np)
        axes[idx][0].set_title("Original Image")
        axes[idx][0].axis('off')

        axes[idx][1].imshow(mask_np, cmap='gray')
        axes[idx][1].set_title("Predicted Mask")
        axes[idx][1].axis('off')

    plt.tight_layout()
    plt.show()

plot_segmented_images()

