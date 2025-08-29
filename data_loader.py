import os
import wfdb
import ast
import pandas as pd

class PTBXLLoader:
    def __init__(self, csv_path, base_path):
        self.df = pd.read_csv(csv_path)
        self.base_path = base_path

    def load_record(self, row, use_high_res=False):
        # Use high or low resolution
        fname = row['filename_hr'] if use_high_res else row['filename_lr']

        # Strip leading 'records100/' if present in the CSV
        if fname.startswith('records100/'):
            fname = fname[len('records100/'):]  # remove first 10 characters

        # Join with base_path
        record_path = os.path.join(self.base_path, fname).replace('.dat','')
        record = wfdb.rdrecord(record_path)
        X = record.p_signal
        y = ast.literal_eval(row['scp_codes'])
        return X, y

    def get_dataframe(self):
        return self.df
