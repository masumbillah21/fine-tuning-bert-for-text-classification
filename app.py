from data_pipeline import DataPipeline
from model_trainer import ModelTrainer

data_pipeline = DataPipeline(file_path='./data/bbc_news.csv', text_column='description', label_column='label')
train_loader, val_loader, test_loader, num_labels = data_pipeline.load_and_preprocess()

trainer = ModelTrainer(num_labels=num_labels)
trainer.train(train_loader, val_loader)
trainer.evaluate(test_loader)
trainer.save_model(data_pipeline.tokenizer)

data_pipeline.save_label_encoder(trainer.save_directory)