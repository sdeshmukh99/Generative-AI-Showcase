# Fine-Tuning GPT-2 on MedQuAD for Medical Question Answering

This project demonstrates the fine-tuning of GPT-2 for the Medical Question Answering (MedQuAD) dataset. The goal is to adapt GPT-2 to answer medical questions effectively.

## Key Steps:
1. Installing dependencies.
2. Preparing the dataset (MedQuAD).
3. Fine-tuning GPT-2 on the dataset.
4. Evaluating the model performance.

## Code Walk-through

The code performs the following steps:

### 01. Installing Dependencies

- Necessary packages are installed using `pip`, including `pyarrow`, `datasets`, `accelerate`, and `transformers`.
  
### 02. Importing Required Packages

- All essential libraries are imported, including PyTorch, Hugging Face Transformers, and others.
- Warnings are suppressed to keep the output clean.
- The `model_output_path` is defined for saving the model.

### 03. Downloading the Dataset

- The MedQuAD dataset is downloaded and loaded into a pandas DataFrame.

### 04. Data Preprocessing

- Missing values are handled by dropping rows with missing 'Question' or 'Answer' fields.
- Duplicates are removed based on the 'Question' and 'Answer' columns to improve data quality.
- The total number of categories in the 'Focus' column is calculated.
- The top 100 focus categories are identified for further processing.

### 05. Creating Training and Validation Sets

- Data is split into training and validation sets per focus category.
- The number of samples per category is reduced to speed up training, which is practical given resource constraints.

### 06. Preparing Data for Model Input

- Questions and answers are combined with special tokens (`<question>`, `<answer>`, `<end>`) to create a suitable input format for the model.
- The combined data is saved into text files for both training and validation.

### 07. Tokenizer and Model Setup

- The GPT-2 tokenizer is loaded, and special tokens are added successfully.
- The `pad_token` is set to the `eos_token`, which is necessary for proper input handling.
- The GPT-2 model is loaded, and its embeddings are resized to match the updated tokenizer.
- Gradient checkpointing is enabled to reduce memory usage during training.

### 08. Tokenizing the Data

- The datasets are loaded from the text files and tokenized using the updated tokenizer.
- The `max_length` parameter is set to 256 to manage memory usage and training time.

### 09. Data Collator Creation

- A data collator is created for language modeling, which helps in dynamically padding inputs during batching.

### 10. Fine-Tuning the GPT-2 Model

- Training arguments are set up correctly, including early stopping to prevent overfitting.
- The model is trained using the `Trainer` class from Hugging Face Transformers.
- Both the model and tokenizer are saved properly.
- The contents of the model directory are displayed to verify the saved files.

### 11. Testing the Fine-Tuned Model

- The fine-tuned model and tokenizer are loaded correctly.
- A `generate_response` function is defined to generate responses to prompts.
- The model is tested with sample prompts, and outputs are printed.

### 12. Comparing with the Original GPT-2 Model

- The original GPT-2 model and tokenizer are loaded without fine-tuning.
- Both models are tested with the same prompts to compare their responses, highlighting the improvements from fine-tuning.

## Files

- **Fine-Tuning GPT-2 on MedQuAD for Medical Question Answering.ipynb**: The main Colab notebook containing the fine-tuning code and workflow.

## Business Applications

A fine-tuned model such as this can be used in healthcare to provide automated responses to medical queries, improving patient care and reducing the workload for medical professionals.
