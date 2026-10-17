"""Chargement du jeu de données patients."""
import pandas as pd
from utils.preprocessing import train_test_split


class Dataset:
    def __init__(self, csv_path, target="infecte"):
        self.csv_path = csv_path
        self.target = target

    def load(self):
        df = pd.read_csv(self.csv_path)
        self.feature_names = [c for c in df.columns if c != self.target]
        self.X = df[self.feature_names].values.astype(float)
        self.y = df[self.target].values.astype(int)
        return self

    def split(self, test_size=0.2, seed=0):
        return train_test_split(self.X, self.y, test_size, seed)
