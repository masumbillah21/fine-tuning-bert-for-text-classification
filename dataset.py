import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer

class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        texts = [str(text) for text in texts]
        self.encodings = tokenizer(
            texts,
            truncation=True,
            padding='max_length',
            max_length=max_length,
            return_tensors='pt'
        )

        labels = [int(label) if str(label).replace('.', '', 1).isdigit() else label for label in labels]  # Handle numeric strings
        self.labels = torch.tensor(labels, dtype=torch.long)

        print(f"TextDataset labels dtype: {self.labels.dtype}")
        print(f"TextDataset labels sample: {self.labels[:5]}")

    def __getitem__(self, idx):
        item = {key: self.encodings[key][idx] for key in self.encodings}
        item['labels'] = self.labels[idx]
        return item

    def __len__(self):
        return len(self.labels)