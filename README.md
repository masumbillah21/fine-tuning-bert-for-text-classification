# BERT Fine-Tuning for Text Classification

This project fine-tunes a BERT model (`bert-base-uncased`) for multi-class text classification on the BBC News dataset (`bbc_news.csv`). The dataset is expected to have columns `title`, `pubDate`, `guid`, `link`, `description`, and `label`, where `description` is used as input text and `label` contains categorical labels (e.g., "World", "Sports", "Business").

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
  - `label` should contain categorical values (e.g., "World", "Sports", "Business").
  - Example row:
    ```
    Ukraine: Angry Zelensky vows to punish Russian atrocities,"Mon, 07 Mar 2022 08:01:56 GMT",https://www.bbc.co.uk/news/world-europe-60638042,https://www.bbc.co.uk/news/world-europe-60638042?at_medium=RSS&at_campaign=KARANGA,The Ukrainian president says the country will not forgive or forget those who murder its civilians.,World
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
   - Ensure `bbc_news.csv` is in the `data` subdirectory (`fine-tuning-bert-for-text-classification\data`).

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

3. **Verify Dataset**:
   - Run the debugging script to inspect `bbc_news.csv`:
     ```bash
     python debug_dataset.py
     ```
   - Check the output for:
     - Columns: Should include `title`, `pubDate`, `guid`, `link`, `description`, `label`.
     - `Label column dtype`: Should be `object` or `string`.
     - `Unique labels`: Should list multiple categories (e.g., `['World', 'Sports', 'Business']`).
     - `Number of unique labels`: Should be > 1 for multi-class classification.
   - If the dataset has only one unique label or contains floats, preprocess it:
     ```python
     import pandas as pd
     df = pd.read_csv('./data/bbc_news.csv')
     df['label'] = df['label'].astype(str)  # Ensure categorical labels
     df.to_csv('./data/bbc_news_cleaned.csv', index=False)
     ```
     Update `app.py` to use `bbc_news_cleaned.csv` by changing the file path:
     ```python
     data_pipeline = DataPipeline(file_path='./data/bbc_news_cleaned.csv', text_column='description', label_column='label')
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
  - Prediction, e.g., `Prediction for 'The stock market saw significant gains today as tech companies rallied.': Business`.
- **Custom predictions**:
  - Modify the `sample_text` in `inference.py` to predict on new text:
    ```python
    sample_text = "Your custom text here"
    ```

## Notes
- **Label Handling**:
  - The pipeline assumes `label` contains categorical values (e.g., "World", "Sports"). If labels are numeric (e.g., `1.0`, `2.0`), they are converted to integers.
  - Ensure the dataset has multiple unique labels for multi-class classification.
- **Hyperparameters**:
  - Batch size: 16 (adjust in `data_pipeline.py` if memory issues occur).
  - Epochs: 3 (adjust in `model_trainer.py` for more training).
  - Learning rate: 2e-5 (adjust in `model_trainer.py` for fine-tuning).
- **Model Saving**:
  - The trained model is saved to `./fine_tuned_bert_bbc`. Ensure this directory is writable.
- **GPU Support**:
  - The pipeline automatically uses a GPU if available (`cuda`). If no GPU is available, it defaults to CPU.

## NOTE:
Ensure multiple unique labels (e.g., "World", "Sports", "Business") are present for proper classification.