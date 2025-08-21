# BERT Fine-Tuning for Text Classification

This project fine-tunes a BERT model (`bert-base-uncased`) for multi-class text classification on the BBC News dataset (`bbc_news.csv`). The dataset is expected to have columns `title`, `pubDate`, `guid`, `link`, `description`, and `label`, where `description` is used as input text and `label` contains categorical labels (e.g., "business", "entertainment", "politics", "sport", "tech").

The pipeline is implemented in Python using PyTorch and the Hugging Face `transformers` library, with a modular class-based structure for data preprocessing, model training, and inference.

## Project Structure

- `dataset.py`: Defines the `TextDataset` class for tokenizing text and preparing labels.
- `data_pipeline.py`: Handles data loading, preprocessing, label encoding, and DataLoader creation.
- `model_trainer.py`: Implements the `ModelTrainer` class for training and evaluating the BERT model.
- `inference.py`: Provides the `Inference` class for predicting categories on new text.
- `app.py`: Orchestrates the pipeline to preprocess data, train the model, and save it.
- `data/bbc_news.csv`: The input dataset (ensure it exists in the `data` subdirectory).

## Prerequisites

- **Python**: Version 3.8 or higher.
- **uv**: The `uv` package manager for installing dependencies. Install `uv` if not already installed:

  ```bash
  pip install uv
  ```

- **Dependencies**: Install required libraries using `uv` (see **Setup**).
- **Dataset**: Place `bbc_news.csv` in the `data` subdirectory. The dataset should have:
  - Columns: `title`, `pubDate`, `guid`, `link`, `description`, `label`.
  - `label` should contain categorical values (e.g., "business", "entertainment", "politics", "sport", "tech").
  - Example row:

    ```
    "TechCorp Stocks Surge After Earnings Report","2025-01-01 08:15:00",,"","TechCorp reported a 15% increase in quarterly profits, boosting shares.","business"
    ```

- **Hardware**: A GPU is recommended for faster training, but CPU is supported.

## Setup

1. **Clone or Create the Project Directory**:
   - Create a directory (e.g., `fine-tuning-bert-for-text-classification`).
   - Save the following files in the directory:
     - `dataset.py`
     - `data_pipeline.py`
     - `model_trainer.py`
     - `inference.py`
     - `app.py`
   - Ensure `bbc_news.csv` is in the `data` subdirectory (`fine-tuning-bert-for-text-classification/data`).

2. **Install Dependencies with uv**:
   - Open a terminal or command prompt.
   - Navigate to the project directory:

     ```bash
     cd fine-tuning-bert-for-text-classification
     ```

   - Install required libraries with specific versions:

     ```bash
     uv sync
     ```

## Usage

### 1. Train the Model

- Run the main script to preprocess the dataset, train the BERT model, and save it:

  ```bash
  uv run app.py
  ```

- **What it does**:
  - Loads and preprocesses `bbc_news.csv` using `DataPipeline`.
  - Splits data into training (70%), validation (15%), and test (15%) sets.
  - Fine-tunes `bert-base-uncased` for 3 epochs.
  - Evaluates the model on the test set.
  - Saves the model, tokenizer, and label encoder to `./fine_tuned_bert_bbc`.
- **Expected output**:
  - Debug prints showing dataset info, label dtype (`torch.long`), and number of labels (should be > 1).
  - Training progress with loss and accuracy for each epoch.
  - Test accuracy.
  - Confirmation that the model is saved: `Model and tokenizer saved to ./fine_tuned_bert_bbc`.

### 2. Perform Inference

- Use the trained model to predict categories for new text:

  ```bash
  uv run inference.py
  ```

- **What it does**:
  - Loads the saved model, tokenizer, and label encoder from `./fine_tuned_bert_bbc`.
  - Predicts the category for a sample text (e.g., "The stock market saw significant gains today as tech companies rallied.").
- **Expected output**:
  - Prediction, e.g., `Prediction for 'The stock market saw significant gains today as tech companies rallied.': business`.

## Notes

- **Label Handling**:
  - The pipeline assumes `label` contains categorical values (e.g., "business", "entertainment"). If labels are numeric (e.g., `1.0`, `2.0`), they are converted to integers.
  - Ensure the dataset has multiple unique labels for multi-class classification.
- **Hyperparameters**:
  - Batch size: 16 (adjust in `data_pipeline.py` if memory issues occur).
  - Epochs: 3 (adjust in `model_trainer.py` for more training).
  - Learning rate: 2e-5 (adjust in `model_trainer.py` for fine-tuning).
- **Model Saving**:
  - The trained model is saved to `./fine_tuned_bert_bbc`. Ensure this directory is writable.
- **GPU Support**:
  - The pipeline automatically uses a GPU if available (`cuda`). If no GPU is available, it defaults to CPU.

## Training Results

The model was trained on a dataset with 5 labels (`business`, `entertainment`, `politics`, `sport`, `tech`). Below are the training results:

```
Model initialized with 5 labels
Epoch 1/3
Train Loss: 1.3694
Validation Loss: 1.1088
Validation Accuracy: 0.7800

Epoch 2/3
Train Loss: 0.9517
Validation Loss: 0.8309
Validation Accuracy: 0.9600

Epoch 3/3
Train Loss: 0.7512
Validation Loss: 0.7498
Validation Accuracy: 0.9600

Test Accuracy: 0.9804
Model and tokenizer saved to ./fine_tuned_bert_bbc
```

## Notes on Results

- The model achieved a test accuracy of 98.04%, indicating strong performance on the test set.
- Validation accuracy improved from 78.00% to 96.00% over 3 epochs, showing effective learning.
- The decrease in train and validation loss suggests the model is converging well.
- Ensure the dataset (`bbc_news.csv`) contains sufficient data (e.g., ~400 rows with balanced labels) to avoid overfitting, as small datasets may lead to high test accuracy but poor generalization.