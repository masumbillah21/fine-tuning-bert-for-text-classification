from transformers import BertTokenizerFast
from data_prep import prepare_data
from model import create_model
from trainer import Trainer
from utils import save_model, load_model_for_inference

if __name__ == "__main__":
    file_path = "news_dataset.csv"

    tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
    train_loader, val_loader, label_names = prepare_data(file_path, tokenizer)

    model, optimizer, scheduler, device = create_model(num_labels=len(label_names), train_loader=train_loader)

    trainer = Trainer(model, optimizer, scheduler, device)
    trainer.train(train_loader, val_loader, num_epochs=3)

    save_model(model, tokenizer, label_names)

    predictor = load_model_for_inference()

    print(predictor(["The stock market is falling rapidly."]))
    print(predictor(["The new iPhone release caused excitement among fans."]))
    print(predictor(["The football match ended in a thrilling draw."]))
