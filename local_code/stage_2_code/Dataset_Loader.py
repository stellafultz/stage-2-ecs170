import pandas as pd
import os

class Dataset_Loader:
    def __init__(self, dName=None, dDescription=None):
        self.dataset_name = dName
        self.dataset_description = dDescription
        self.dataset_source_folder_path = None
        self.dataset_source_file_name = None

    def load(self):
        # __file__ is local_code/stage_2_code/Dataset_Loader.py
        # go up 2 levels to reach project root
        base = os.path.dirname(os.path.abspath(__file__))  # .../local_code/stage_2_code
        project_root = os.path.join(base, '..', '..')      # .../project root
        full_path = os.path.normpath(os.path.join(
            project_root,
            self.dataset_source_folder_path,
            self.dataset_source_file_name
        ))
        df = pd.read_csv(full_path, header=None)
        X = df.iloc[:, 1:].values / 255.0
        y = df.iloc[:, 0].values
        return X, y