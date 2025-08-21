import torch
from tqdm import tqdm
from sklearn.metrics import accuracy_score
from evaluator import evaluate_model

class Trainer:
    def __init__(self, model, optimizer, scheduler, device):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device

    def train(self, train_loader, val_loader, num_epochs=3):
        for epoch in range(num_epochs):
            self.model.train()
            train_loss, train_preds, train_labels = 0, [], []

            loop = tqdm(train_loader, desc=f"Epoch {epoch+1}")
            for batch in loop:
                self.optimizer.zero_grad()

                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["label"].to(self.device)

                outputs = self.model(input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                logits = outputs.logits

                loss.backward()
                self.optimizer.step()
                self.scheduler.step()

                train_loss += loss.item()
                train_preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
                train_labels.extend(labels.cpu().numpy())

            train_acc = accuracy_score(train_labels, train_preds)
            print(f"Epoch {epoch+1} | Train Loss: {train_loss/len(train_loader):.4f}, Train Acc: {train_acc:.4f}")

            evaluate_model(self.model, val_loader, self.device, split="Validation")
