from data_loader import PTBXLLoader
from trainer import train_model
from utils import get_logger
import torch
import ast

logger = get_logger()

# Use the same 5 diagnoses
DIAGNOSES = ['NORM', 'MI', 'STTC', 'CD', 'HYP']

def run_pipeline():
    csv_path = r"C:\Users\15166\OneDrive\Desktop\ekg_ai_project\ptbxl_database.csv"
    records_path = r"C:\Users\15166\OneDrive\Desktop\ekg_ai_project\records100"

    loader = PTBXLLoader(csv_path=csv_path, base_path=records_path)
    df = loader.get_dataframe()

    X_list, y_list = [], []
    label_map = {code: idx for idx, code in enumerate(DIAGNOSES)}

    # Load first 50 records for demo/training (increase for full training)
    for i in range(50):
        X, y_dict = loader.load_record(df.iloc[i])
        y = [0]*len(DIAGNOSES)
        for code in y_dict.keys():
            if code in label_map:
                y[label_map[code]] = 1
        X_list.append(X)
        y_list.append(y)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = train_model(X_list, y_list, device=device)

if __name__ == "__main__":
    run_pipeline()
