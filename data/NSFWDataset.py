from torch.utils.data import Dataset
import pandas as pd
import torch
from transformers import AutoTokenizer


class NSFWDataset(Dataset):
    def __init__(self, csv_path, max_length=64, mode="train"):
        self.max_length = max_length
        self.data = pd.read_csv(csv_path)
        self.data = self.data[["title", "over_18"]]
        if mode == "train":
            trues = self.data[self.data["over_18"] == True]
            falses = self.data[self.data["over_18"] == False].sample(
                n=len(trues), random_state=42
            )
            self.data = pd.concat([trues, falses])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        title = row["title"]
        label = int(row["over_18"])
        return title, label


if __name__ == "__main__":
    dataset = NSFWDataset("../dataset/r_dataisbeautiful_posts.csv")
    title, label = dataset[0]
    print(title, label)
