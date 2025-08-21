import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

def evaluate_model(model, data_loader, device, split="Test"):
    model.eval()
    preds, labels = [], []
    total_loss = 0

    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            label = batch["label"].to(device)

            outputs = model(input_ids, attention_mask=attention_mask, labels=label)
            total_loss += outputs.loss.item()

            preds.extend(torch.argmax(outputs.logits, dim=1).cpu().numpy())
            labels.extend(label.cpu().numpy())

    acc = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="weighted")

    print(f"{split} Loss: {total_loss/len(data_loader):.4f}, Acc: {acc:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
    return acc, precision, recall, f1
