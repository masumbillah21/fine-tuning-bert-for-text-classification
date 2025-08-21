import torch
from transformers import BertForSequenceClassification, AdamW, get_scheduler

def create_model(num_labels, lr=2e-5, train_loader=None, num_epochs=3):
    model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=num_labels)
    optimizer = AdamW(model.parameters(), lr=lr)

    num_training_steps = len(train_loader) * num_epochs if train_loader else 1000
    scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    return model, optimizer, scheduler, device
