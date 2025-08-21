import torch
import pandas as pd
from torch.utils.data import Dataset
from sklearn.preprocessing import LabelEncoder

class TextDataset(Dataset):
    def __init__(self, file_path, tokenizer, max_len=128):
        df = pd.read_csv(file_path)

        self.label_encoder = LabelEncoder()
        self.labels = self.label_encoder.fit_transform(df["label"])
        self.label_names = list(self.label_encoder.classes_)

        encodings = tokenizer(
            list(df["text"]), padding="max_length", truncation=True, max_length=max_len
        )
        self.encodings = encodings

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["label"] = torch.tensor(self.labels[idx])
        return item
