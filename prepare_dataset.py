# prepare_dataset.py
# -*- coding: utf-8 -*-
"""
Prepare PTB-XL metadata splits + labels (5 superclasses) without loading signals.
- Inputs: ptbxl_database.csv, scp_statements.csv, records100/ folder
- Outputs: prepared/{train.csv,val.csv,test.csv,class_counts.json}
Each CSV contains:
  ecg_id, record_path (base path for WFDB), labels_json, label_NORM, label_MI, label_STTC, label_HYP, label_CD, primary_class
This keeps data engineering decoupled from model I/O (fits mentor’s guidance).
"""

import os
import json
import ast
import argparse
import pandas as pd
from sklearn.model_selection import train_test_split

SUPERCLASSES = ["NORM", "MI", "STTC", "HYP", "CD"]

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", required=True, help="Path to ptbxl_database.csv")
    p.add_argument("--scp", required=True, help="Path to scp_statements.csv")
    p.add_argument("--records", required=True, help="Base folder to records100")
    p.add_argument("--out", required=True, help="Output folder for prepared files")
    p.add_argument("--val_size", type=float, default=0.1, help="Validation fraction (of total)")
    p.add_argument("--test_size", type=float, default=0.1, help="Test fraction (of total)")
    p.add_argument("--seed", type=int, default=42, help="Random seed")
    return p.parse_args()

def load_metadata(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    # scp_codes is a stringified dict: convert to dict
    df["scp_codes"] = df["scp_codes"].apply(ast.literal_eval)
    return df

def load_scp_mapping(scp_path: str) -> dict:
    """Return dict: SCP code -> diagnostic_superclass (only diagnostic==1)."""
    scp = pd.read_csv(scp_path, index_col=0)
    # Keep diagnostic codes only if column exists; otherwise assume all rows are diagnostic
    if "diagnostic" in scp.columns:
        scp = scp[scp["diagnostic"] == 1]
    if "diagnostic_class" not in scp.columns:
        raise ValueError("scp_statements.csv must contain a 'diagnostic_class' column.")
    return scp["diagnostic_class"].to_dict()

def record_base_path(records_root: str, filename_lr: str) -> str:
    """
    Build WFDB base path for rdrecord.
    - Strips leading 'records100/' if present (CSV may already include it).
    - Strips a trailing extension like '.dat' if present.
    """
    rel = filename_lr.replace("\\", "/")
    if rel.startswith("records100/"):
        rel = rel[len("records100/"):]
    if rel.lower().endswith(".dat"):
        rel = rel[:-4]
    return os.path.normpath(os.path.join(records_root, rel))

def labels_from_scp_codes(scp_codes: dict, code_to_super: dict) -> list:
    """Return sorted unique superclasses present for a row."""
    present = set()
    for code in scp_codes.keys():
        cls = code_to_super.get(code)
        if cls in SUPERCLASSES:
            present.add(cls)
    return sorted(present)

def one_hot(classes_present: list) -> dict:
    return {f"label_{c}": (1 if c in classes_present else 0) for c in SUPERCLASSES}

def pick_primary_class(classes_present: list) -> str:
    """
    Choose a single primary class for stratification & quick summaries.
    Heuristic:
      - if only one, use it
      - else prefer any non-NORM; otherwise first alphabetically
    """
    if not classes_present:
        return "NONE"
    if len(classes_present) == 1:
        return classes_present[0]
    non_norm = [c for c in classes_present if c != "NORM"]
    return non_norm[0] if non_norm else classes_present[0]

def main():
    args = parse_args()
    os.makedirs(args.out, exist_ok=True)

    print("[INFO] Loading metadata...")
    df = load_metadata(args.csv)

    print("[INFO] Loading SCP mapping...")
    code_to_super = load_scp_mapping(args.scp)

    print("[INFO] Building rows with labels and record paths...")
    rows = []
    class_counts = {c: 0 for c in SUPERCLASSES}
    with_any = 0
    for _, row in df.iterrows():
        labels = labels_from_scp_codes(row["scp_codes"], code_to_super)
        if labels:
            with_any += 1
            for c in labels:
                class_counts[c] += 1

        base_path = record_base_path(args.records, row["filename_lr"])
        oh = one_hot(labels)
        rows.append({
            "ecg_id": int(row["ecg_id"]),
            "record_path": base_path,               # pass this to wfdb.rdrecord(record_path)
            "labels_json": json.dumps(labels),      # multi-label truth (json list of superclasses)
            **oh,
            "primary_class": pick_primary_class(labels)
        })

    prepared = pd.DataFrame(rows)

    # Keep rows that have at least one diagnostic superclass
    prepared = prepared[prepared["primary_class"] != "NONE"].reset_index(drop=True)

    # Print counts
    total_rows = len(df)
    print(f"[INFO] Total rows in CSV: {total_rows}")
    print(f"[INFO] Rows with at least one diagnostic superclass: {with_any}")
    print("[INFO] Class counts (multi-label presence):")
    for c in SUPERCLASSES:
        print(f"  {c}: {class_counts[c]}")
    # Save counts for instrumentation/automation
    with open(os.path.join(args.out, "class_counts.json"), "w", encoding="utf-8") as f:
        json.dump({
            "total_rows": total_rows,
            "with_any_superclass": with_any,
            "class_counts": class_counts
        }, f, indent=2)

    # Train/Val/Test split (stratify by primary_class for stability)
    test_size = args.test_size
    val_size = args.val_size
    if test_size + val_size >= 0.9:
        raise ValueError("test_size + val_size should be less than 0.9")

    train_df, temp_df = train_test_split(
        prepared,
        test_size=(test_size + val_size),
        random_state=args.seed,
        stratify=prepared["primary_class"]
    )
    relative_val = val_size / (test_size + val_size) if (test_size + val_size) > 0 else 0.5
    val_df, test_df = train_test_split(
        temp_df,
        test_size=(1 - relative_val),
        random_state=args.seed,
        stratify=temp_df["primary_class"]
    )

    # Save splits
    train_path = os.path.join(args.out, "train.csv")
    val_path   = os.path.join(args.out, "val.csv")
    test_path  = os.path.join(args.out, "test.csv")

    cols = ["ecg_id", "record_path", "labels_json"] + [f"label_{c}" for c in SUPERCLASSES] + ["primary_class"]
    train_df[cols].to_csv(train_path, index=False)
    val_df[cols].to_csv(val_path, index=False)
    test_df[cols].to_csv(test_path, index=False)

    print("[INFO] Saved:")
    print(f"  {train_path}  ({len(train_df)} rows)")
    print(f"  {val_path}    ({len(val_df)} rows)")
    print(f"  {test_path}   ({len(test_df)} rows)")
    print("✅ Done.")

if __name__ == "__main__":
    main()
