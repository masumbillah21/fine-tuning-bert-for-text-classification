import torch
import pickle
from transformers import BertTokenizer, BertForSequenceClassification

class Inference:
    def __init__(self, model_path="./fine_tuned_bert_bbc"):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = BertForSequenceClassification.from_pretrained(model_path)
        self.tokenizer = BertTokenizer.from_pretrained(model_path)
        with open(f"{model_path}/label_encoder.pkl", 'rb') as f:
            self.label_encoder = pickle.load(f)
        self.model.to(self.device)
        self.model.eval()

    def predict_category(self, text, max_length=128):
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=max_length
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self.model(**inputs)
        logits = outputs.logits
        prediction = torch.argmax(logits, dim=1).item()
        return self.label_encoder.inverse_transform([prediction])[0]

if __name__ == "__main__":
    inference = Inference()
    sample_text = "The stock market saw significant gains today as tech companies rallied."
    prediction = inference.predict_category(sample_text)
    print(f"Prediction for '{sample_text}': {prediction}")