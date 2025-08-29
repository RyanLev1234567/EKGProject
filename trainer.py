import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from data_loader import PTBXLLoader
from dataset_builder import ECGDataset
from model import CNN_LSTM_Attention
from preprocess import normalize_signal, resample_signal
from tqdm import tqdm
import numpy as np

# Major 5 diagnoses
DIAGNOSES = ['NORM', 'MI', 'STTC', 'CD', 'HYP']

def build_dataset(csv_path, records_path):
    loader = PTBXLLoader(csv_path=csv_path, base_path=records_path)
    df = loader.get_dataframe()

    X_list, y_list = [], []

    print("Loading and processing records...")
    for idx in tqdm(range(len(df)), desc="Processing records"):
        try:
            X, y_dict = loader.load_record(df.iloc[idx])
        except FileNotFoundError:
            continue

        y = [0]*len(DIAGNOSES)
        for code in y_dict.keys():
            if code in DIAGNOSES and y_dict[code] > 0:  # presence threshold
                y[DIAGNOSES.index(code)] = 1

        X = normalize_signal(resample_signal(X))
        X_list.append(torch.tensor(X, dtype=torch.float))
        y_list.append(torch.tensor(y, dtype=torch.float))

    return ECGDataset(X_list, y_list)

def train_model(csv_path, records_path, device='cpu', num_epochs=5, batch_size=32, learning_rate=1e-3, model_save_path="cnn_lstm_attention_5diagnoses.pth"):
    dataset = build_dataset(csv_path, records_path)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = CNN_LSTM_Attention(input_channels=12, num_classes=len(DIAGNOSES)).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    print("Starting training...")
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for X_batch, y_batch in tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * X_batch.size(0)

        epoch_loss = running_loss / len(dataset)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}")

    # Save trained model
    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved as {model_save_path}")

if __name__ == "__main__":
    csv_path = r"C:\Users\15166\OneDrive\Desktop\ekg_ai_project\ptbxl_database.csv"
    records_path = r"C:\Users\15166\OneDrive\Desktop\ekg_ai_project\records100"
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    train_model(csv_path, records_path, device=device, num_epochs=5, batch_size=32)
