import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader
from transformers import BertTokenizer
from dataset import TextDataset

class DataPipeline:
    def __init__(self, file_path, text_column='description', label_column='label', max_length=128, batch_size=16):
        self.file_path = file_path
        self.text_column = text_column
        self.label_column = label_column
        self.max_length = max_length
        self.batch_size = batch_size
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.label_encoder = LabelEncoder()
        self.data = None
        self.labels = None
        self.num_labels = None

    def load_and_preprocess(self):
        self.data = pd.read_csv(self.file_path)
        
        print("Dataset info:")
        print(f"Columns: {self.data.columns.tolist()}")
        print(f"Shape: {self.data.shape}")
        print(f"Missing values:\n{self.data[[self.text_column, self.label_column]].isnull().sum()}")
        print(f"Label column dtype: {self.data[self.label_column].dtype}")
        print(f"Unique labels: {self.data[self.label_column].unique()}")
        
        self.data[self.text_column] = self.data[self.text_column].astype(str).fillna('')
        
        self.data[self.label_column] = self.data[self.label_column].astype(str)
        
        self.labels = self.label_encoder.fit_transform(self.data[self.label_column])
        self.num_labels = len(self.label_encoder.classes_)
        if self.num_labels <= 1:
            raise ValueError(f"Only {self.num_labels} unique label(s) found. Expected multiple labels for classification.")
        print(f"Number of labels: {self.num_labels}")
        print(f"Encoded labels dtype: {self.labels.dtype}")
        print(f"Encoded labels sample: {self.labels[:5]}")
        print(f"Label classes: {self.label_encoder.classes_}")

        texts = self.data[self.text_column].tolist()
        
        train_texts, temp_texts, train_labels, temp_labels = train_test_split(
            texts, self.labels, test_size=0.3, random_state=42
        )
        val_texts, test_texts, val_labels, test_labels = train_test_split(
            temp_texts, temp_labels, test_size=0.5, random_state=42
        )
        
        train_labels = np.array(train_labels, dtype=np.int64)
        val_labels = np.array(val_labels, dtype=np.int64)
        test_labels = np.array(test_labels, dtype=np.int64)
        
        print(f"Train labels dtype: {train_labels.dtype}")
        print(f"Train labels sample: {train_labels[:5]}")
        
        self.train_dataset = TextDataset(train_texts, train_labels, self.tokenizer, self.max_length)
        self.val_dataset = TextDataset(val_texts, val_labels, self.tokenizer, self.max_length)
        self.test_dataset = TextDataset(test_texts, test_labels, self.tokenizer, self.max_length)
        
        self.train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)
        self.val_loader = DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False)
        self.test_loader = DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False)
        
        for batch in self.train_loader:
            print(f"Batch labels dtype: {batch['labels'].dtype}")
            print(f"Batch labels sample: {batch['labels'][:5]}")
            break
        
        return self.train_loader, self.val_loader, self.test_loader, self.num_labels

    def save_label_encoder(self, save_directory):
        import pickle
        with open(f"{save_directory}/label_encoder.pkl", 'wb') as f:
            pickle.dump(self.label_encoder, f)