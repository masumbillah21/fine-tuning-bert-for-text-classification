import pandas as pd
import torch
from transformers import BertForSequenceClassification, BertTokenizerFast

def save_model(model, tokenizer, label_names, save_dir="./bert_finetuned"):
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)

    pd.Series(label_names).to_csv(f"{save_dir}/labels.csv", index=False, header=False)
    print(f"Model & labels saved to {save_dir}")

def load_model_for_inference(save_dir="./bert_finetuned"):
    model = BertForSequenceClassification.from_pretrained(save_dir)
    tokenizer = BertTokenizerFast.from_pretrained(save_dir)

    label_names = pd.read_csv(f"{save_dir}/labels.csv", header=None)[0].tolist()

    def predict(texts):
        model.eval()
        encodings = tokenizer(texts, padding=True, truncation=True, max_length=128, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**encodings)
            preds = torch.argmax(outputs.logits, dim=1).cpu().numpy()
        return [label_names[p] for p in preds]

    return predict
