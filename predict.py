import torch
import matplotlib.pyplot as plt
from model import UNet
from torch.utils.data import DataLoader, TensorDataset
from config import device
import os
import torchvision.transforms as transforms 
from PIL import Image

device = device
model = UNet(3,1)
path = os.path.join(os.getcwd(), 'checkpoints', 'unet_breast_cancer_segmentation.pth')
model.load_state_dict(torch.load(path, map_location=device))
model = model.to(device)
model.eval()

def plot_segmented_images(image_paths):
    if isinstance(image_paths, str):
        image_paths = [image_paths]
    image_wo_mask = []
    for filename in image_paths:
        if '_mask' not in filename and filename.endswith(('.png', '.jpg', '.jpeg')):
            image_wo_mask.append(filename)
    max_plots = 5
    image_wo_mask = image_wo_mask[:max_plots]
    print(image_wo_mask)
    if not image_wo_mask:
        print("No valid image files found.")
        return
    print(f"Processing {len(image_wo_mask)} images")

    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    images = []
    for path in image_wo_mask:
        img = Image.open(path).convert('RGB')
        img = transform(img)
        images.append(img)

    images = torch.stack(images)  
    dataset = TensorDataset(images)  
    loader = DataLoader(dataset, len(images), shuffle=False)

    batch = next(iter(loader))[0].to(device)

    model.eval()
    with torch.no_grad():
        outputs = model(batch)
        outputs = torch.sigmoid(outputs)
        preds = (outputs > 0.5).float()

    batch = batch.cpu()
    preds = preds.cpu()
    fig, axes = plt.subplots(len(batch), 2, figsize=(8, 4 * len(batch)))

    if len(batch) == 1:
        axes = [axes]

    for idx, (img, pred) in enumerate(zip(batch, preds)):
        img = img * 0.5 + 0.5  # Unnormalize
        img = img.permute(1, 2, 0).numpy()

        pred = pred.squeeze().numpy()

        axes[idx][0].imshow(img)
        axes[idx][0].set_title("Original Image")
        axes[idx][0].axis('off')

        axes[idx][1].imshow(pred, cmap='gray')
        axes[idx][1].set_title("Predicted Mask")
        axes[idx][1].axis('off')

    plt.tight_layout()
    plt.show()

