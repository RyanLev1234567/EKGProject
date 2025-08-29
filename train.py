import torch
import torch.nn as nn
import torch.optim as optim

def train_model(model, dataloader, epochs=5, lr=1e-3):
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        for X, y in dataloader:
            optimizer.zero_grad()
            outputs = model(X.float())
            loss = criterion(outputs, y.float())
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch+1}/{epochs} - Loss: {loss.item():.4f}")
