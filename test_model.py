# test_model_comprehensive.py (fixed version)
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import json
import wfdb
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from model import CNN_LSTM_Attention
from preprocess import normalize_signal, resample_signal
import ast

# Define the same diagnoses used during training
DIAGNOSES = ['NORM', 'MI', 'STTC', 'CD', 'HYP']

class ECGTestDataset(Dataset):
    def __init__(self, csv_path, records_path):
        self.df = pd.read_csv(csv_path)
        self.records_path = records_path
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        record_path = row['record_path']
        
        # Load ECG signal
        record = wfdb.rdrecord(record_path)
        X = record.p_signal
        
        # Preprocess the signal (same as training)
        X = normalize_signal(resample_signal(X))
        X = torch.tensor(X, dtype=torch.float)
        
        # Get labels
        labels = json.loads(row['labels_json'])
        y = [1 if diag in labels else 0 for diag in DIAGNOSES]
        y = torch.tensor(y, dtype=torch.float)
        
        return X, y, row['ecg_id']

def test_model(model_path, test_csv_path, records_path, device='cpu', batch_size=32):
    # Load model
    model = CNN_LSTM_Attention(input_channels=12, num_classes=len(DIAGNOSES))
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    
    # Create test dataset and dataloader
    test_dataset = ECGTestDataset(test_csv_path, records_path)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Initialize lists to store results
    all_preds = []
    all_probs = []
    all_labels = []
    all_ecg_ids = []
    
    # Test the model
    with torch.no_grad():
        for X_batch, y_batch, ecg_ids in test_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            outputs = model(X_batch)
            probs = torch.sigmoid(outputs)
            preds = (probs > 0.5).int()
            
            all_probs.append(probs.cpu().numpy())
            all_preds.append(preds.cpu().numpy())
            all_labels.append(y_batch.cpu().numpy())
            all_ecg_ids.extend(ecg_ids)
    
    # Concatenate all results
    all_probs = np.concatenate(all_probs, axis=0)
    all_preds = np.concatenate(all_preds, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    
    return all_probs, all_preds, all_labels, all_ecg_ids

def evaluate_performance(all_labels, all_preds, all_probs):
    # Calculate metrics for each class
    results = {}
    
    for i, diagnosis in enumerate(DIAGNOSES):
        # Skip if no positive samples in test set
        if np.sum(all_labels[:, i]) == 0:
            print(f"Warning: No positive samples for {diagnosis} in test set")
            results[diagnosis] = {
                'auc': float('nan'),
                'precision': 0.0,
                'recall': 0.0,
                'f1-score': 0.0,
                'support': 0,
                'confusion_matrix': np.array([[0, 0], [0, 0]])
            }
            continue
            
        # Calculate AUC-ROC
        try:
            auc = roc_auc_score(all_labels[:, i], all_probs[:, i])
        except ValueError:
            auc = float('nan')
            
        # Get classification report for this class
        report = classification_report(
            all_labels[:, i], 
            all_preds[:, i], 
            output_dict=True,
            zero_division=0
        )
        
        # Handle case where no positive predictions were made
        if '1' not in report:
            precision = 0.0
            recall = 0.0
            f1 = 0.0
            support = int(np.sum(all_labels[:, i]))
            cm = confusion_matrix(all_labels[:, i], all_preds[:, i])
            if cm.size == 1:  # Only one class present
                if all_labels[0, i] == 0:
                    cm = np.array([[len(all_labels), 0], [0, 0]])
                else:
                    cm = np.array([[0, 0], [len(all_labels), 0]])
        else:
            precision = report['1']['precision']
            recall = report['1']['recall']
            f1 = report['1']['f1-score']
            support = report['1']['support']
            cm = confusion_matrix(all_labels[:, i], all_preds[:, i])
        
        results[diagnosis] = {
            'auc': auc,
            'precision': precision,
            'recall': recall,
            'f1-score': f1,
            'support': support,
            'confusion_matrix': cm
        }
    
    # Calculate micro-average metrics
    try:
        micro_auc = roc_auc_score(all_labels.ravel(), all_probs.ravel())
    except ValueError:
        micro_auc = float('nan')
    
    # Calculate overall metrics with robust error handling
    try:
        overall_report = classification_report(
            all_labels, 
            all_preds, 
            target_names=DIAGNOSES,
            output_dict=True,
            zero_division=0
        )
        
        # Calculate accuracy separately to ensure it's always available
        accuracy = accuracy_score(all_labels.ravel(), all_preds.ravel())
        
        results['overall'] = {
            'micro_auc': micro_auc,
            'accuracy': accuracy,
            'macro_avg': overall_report.get('macro avg', {'precision': 0, 'recall': 0, 'f1-score': 0}),
            'weighted_avg': overall_report.get('weighted avg', {'precision': 0, 'recall': 0, 'f1-score': 0})
        }
    except Exception as e:
        print(f"Warning: Error calculating overall metrics: {e}")
        # Fallback to basic metrics
        accuracy = accuracy_score(all_labels.ravel(), all_preds.ravel())
        results['overall'] = {
            'micro_auc': micro_auc,
            'accuracy': accuracy,
            'macro_avg': {'precision': 0, 'recall': 0, 'f1-score': 0},
            'weighted_avg': {'precision': 0, 'recall': 0, 'f1-score': 0}
        }
    
    return results

def plot_confusion_matrices(results, save_path=None):
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.ravel()
    
    for i, diagnosis in enumerate(DIAGNOSES):
        if diagnosis in results:
            cm = results[diagnosis]['confusion_matrix']
            # Ensure we have a 2x2 matrix
            if cm.shape == (1, 1):
                if results[diagnosis]['support'] == 0:  # No positive samples
                    cm = np.array([[len(cm), 0], [0, 0]])
                else:  # Only positive samples
                    cm = np.array([[0, 0], [len(cm), 0]])
            elif cm.shape == (2, 1):  # Missing prediction column
                cm = np.column_stack([cm, [0, 0]])
            elif cm.shape == (1, 2):  # Missing true label row
                cm = np.row_stack([cm, [0, 0]])
            
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[i])
            axes[i].set_title(f'Confusion Matrix - {diagnosis}')
            axes[i].set_xlabel('Predicted')
            axes[i].set_ylabel('Actual')
    
    # Hide empty subplot
    axes[-1].set_visible(False)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()

def print_detailed_report(results):
    print("=" * 60)
    print("MODEL EVALUATION REPORT")
    print("=" * 60)
    
    # Print per-class metrics
    print("\nPER-CLASS METRICS:")
    print("-" * 60)
    for diagnosis in DIAGNOSES:
        if diagnosis in results:
            metrics = results[diagnosis]
            print(f"{diagnosis:5s} | AUC: {metrics['auc']:.3f} | "
                  f"Precision: {metrics['precision']:.3f} | "
                  f"Recall: {metrics['recall']:.3f} | "
                  f"F1: {metrics['f1-score']:.3f} | "
                  f"Support: {metrics['support']}")
    
    # Print overall metrics
    overall = results['overall']
    print("\nOVERALL METRICS:")
    print("-" * 60)
    print(f"Micro AUC: {overall['micro_auc']:.3f}")
    print(f"Accuracy: {overall['accuracy']:.3f}")
    print(f"Macro Precision: {overall['macro_avg']['precision']:.3f}")
    print(f"Macro Recall: {overall['macro_avg']['recall']:.3f}")
    print(f"Macro F1: {overall['macro_avg']['f1-score']:.3f}")
    print(f"Weighted Precision: {overall['weighted_avg']['precision']:.3f}")
    print(f"Weighted Recall: {overall['weighted_avg']['recall']:.3f}")
    print(f"Weighted F1: {overall['weighted_avg']['f1-score']:.3f}")

def save_results_to_csv(all_probs, all_preds, all_labels, all_ecg_ids, save_path):
    results_df = pd.DataFrame({
        'ecg_id': all_ecg_ids
    })
    
    for i, diagnosis in enumerate(DIAGNOSES):
        results_df[f'{diagnosis}_true'] = all_labels[:, i]
        results_df[f'{diagnosis}_pred'] = all_preds[:, i]
        results_df[f'{diagnosis}_prob'] = all_probs[:, i]
    
    results_df.to_csv(save_path, index=False)
    print(f"Detailed results saved to {save_path}")

if __name__ == "__main__":
    # Configuration
    MODEL_PATH = r"C:\Users\15166\OneDrive\Desktop\ekg_ai_project\cnn_lstm_attention_5diagnoses.pth"
    TEST_CSV_PATH = r"C:\Users\15166\OneDrive\Desktop\ekg_ai_project\prepared\test.csv"
    RECORDS_PATH = r"C:\Users\15166\OneDrive\Desktop\ekg_ai_project\records100"
    RESULTS_CSV_PATH = r"C:\Users\15166\OneDrive\Desktop\ekg_ai_project\test_results.csv"
    CM_PLOT_PATH = r"C:\Users\15166\OneDrive\Desktop\ekg_ai_project\confusion_matrices.png"
    
    # Set device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Load class counts for reference
    with open(r"C:\Users\15166\OneDrive\Desktop\ekg_ai_project\prepared\class_counts.json", 'r') as f:
        class_counts = json.load(f)
        print("Class counts in dataset:", class_counts['class_counts'])
    
    # Test the model
    print("Testing model...")
    all_probs, all_preds, all_labels, all_ecg_ids = test_model(
        MODEL_PATH, TEST_CSV_PATH, RECORDS_PATH, device=device
    )
    
    # Evaluate performance
    results = evaluate_performance(all_labels, all_preds, all_probs)
    
    # Print detailed report
    print_detailed_report(results)
    
    # Plot confusion matrices
    plot_confusion_matrices(results, save_path=CM_PLOT_PATH)
    
    # Save detailed results to CSV
    save_results_to_csv(all_probs, all_preds, all_labels, all_ecg_ids, RESULTS_CSV_PATH)
    
    print("\nTesting completed successfully!")