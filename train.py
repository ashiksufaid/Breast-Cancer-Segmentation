import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import math
def train_model(model, num_epochs, train_loader, loss_fn, optimizer, device):
    model = model.to(device)
    train_losses = []
    optimizer = optimizer(model.parameters(), lr = 0.001)
    for epoch in range(num_epochs):
        running_loss = 0.0

        for batch, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False):
            batch, labels = batch.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(batch)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)
        train_losses.append(avg_loss)
        print(f"Epoch [{epoch+1}/{num_epochs}] - Loss: {avg_loss:.4f}")

    return train_losses