from torch.utils.data import DataLoader, random_split
from dataset import TextDataset

def prepare_data(file_path, tokenizer, batch_size=16, max_len=128):
    dataset = TextDataset(file_path, tokenizer, max_len=max_len)

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    return train_loader, val_loader, dataset.label_names
