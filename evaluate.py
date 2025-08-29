from sklearn.metrics import classification_report

def evaluate_model(model, dataloader):
    all_preds, all_labels = [], []
    for X, y in dataloader:
        preds = model(X.float()).detach().numpy()
        all_preds.append(preds)
        all_labels.append(y.numpy())
    # Flatten and report
    return classification_report(
        [item for sublist in all_labels for item in sublist],
        [item for sublist in all_preds for item in sublist] > 0.5
    )
