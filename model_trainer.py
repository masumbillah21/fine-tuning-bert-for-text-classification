import torch
from torch.optim import AdamW
from transformers import BertForSequenceClassification, get_linear_schedule_with_warmup
from sklearn.metrics import accuracy_score
import numpy as np

class ModelTrainer:
    def __init__(self, num_labels, save_directory="./fine_tuned_bert_bbc", epochs=3, learning_rate=2e-5):
        self.num_labels = num_labels
        self.save_directory = save_directory
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if self.num_labels <= 1:
            raise ValueError(f"num_labels must be > 1 for multi-class classification, got {self.num_labels}")
        self.model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=num_labels)
        self.model.to(self.device)
        print(f"Model initialized with {self.num_labels} labels")

    def train(self, train_loader, val_loader):
        optimizer = AdamW(self.model.parameters(), lr=self.learning_rate, eps=1e-8)
        total_steps = len(train_loader) * self.epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=0,
            num_training_steps=total_steps
        )

        for epoch in range(self.epochs):
            self.model.train()
            total_train_loss = 0

            for batch in train_loader:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                if batch['labels'].dtype != torch.long:
                    raise ValueError(f"Expected labels to be torch.long, got {batch['labels'].dtype}")
                outputs = self.model(**batch)
                loss = outputs.loss
                total_train_loss += loss.item()

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            avg_train_loss = total_train_loss / len(train_loader)

            self.model.eval()
            total_val_loss = 0
            val_predictions = []
            val_true_labels = []

            with torch.no_grad():
                for batch in val_loader:
                    batch = {k: v.to(self.device) for k, v in batch.items()}
                    outputs = self.model(**batch)
                    loss = outputs.loss
                    total_val_loss += loss.item()

                    logits = outputs.logits.detach().cpu().numpy()
                    label_ids = batch['labels'].to('cpu').numpy()
                    val_predictions.extend(np.argmax(logits, axis=1).flatten())
                    val_true_labels.extend(label_ids.flatten())

            avg_val_loss = total_val_loss / len(val_loader)
            val_accuracy = accuracy_score(val_true_labels, val_predictions)

            print(f"Epoch {epoch + 1}/{self.epochs}")
            print(f"Train Loss: {avg_train_loss:.4f}")
            print(f"Validation Loss: {avg_val_loss:.4f}")
            print(f"Validation Accuracy: {val_accuracy:.4f}\n")

    def evaluate(self, test_loader):
        self.model.eval()
        test_predictions = []
        test_true_labels = []

        with torch.no_grad():
            for batch in test_loader:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                outputs = self.model(**batch)
                logits = outputs.logits.detach().cpu().numpy()
                label_ids = batch['labels'].to('cpu').numpy()
                test_predictions.extend(np.argmax(logits, axis=1).flatten())
                test_true_labels.extend(label_ids.flatten())

        test_accuracy = accuracy_score(test_true_labels, test_predictions)
        print(f"Test Accuracy: {test_accuracy:.4f}")

    def save_model(self, tokenizer):
        self.model.save_pretrained(self.save_directory)
        tokenizer.save_pretrained(self.save_directory)
        print(f"Model and tokenizer saved to {self.save_directory}")