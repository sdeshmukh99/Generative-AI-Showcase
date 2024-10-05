## Fine-Tuning GPT-2 on MedQuAD for Medical Question Answering

This project demonstrates the fine-tuning of GPT-2 on the Medical Question Answering (MedQuAD) dataset. The goal is to adapt GPT-2 to answer medical questions effectively.

A fine-tuned model such as this can be used in healthcare to provide automated responses to medical queries, improving patient care and reducing the workload for medical professionals.

### Key Steps:
1. Installing dependencies.
2. Preparing the dataset (MedQuAD).
3. Fine-tuning GPT-2 on the dataset.
4. Evaluating the model performance.

### Overview

The code performs the following steps:

#### 1. Installing Dependencies

- All necessary packages are installed using `pip`, ensuring that the environment is set up properly.
- Packages installed include:
  - `pyarrow==15.0.2`
  - `datasets`
  - `accelerate`
  - `transformers`

#### 2. Importing Required Packages

- All essential libraries are imported, including:
  - **Data Handling and Processing:**
    - `os`
    - `re`
    - `numpy`
    - `pandas`
    - `matplotlib.pyplot`
    - `sklearn.model_selection` (`train_test_split`)
    - `torch`
  - **Hugging Face Libraries:**
    - `datasets`
    - `transformers` (various modules)
- Warnings are suppressed to keep the output clean.
- The `model_output_path` is defined correctly for saving the model.

#### 3. Downloading the Dataset

- The MedQuAD dataset is downloaded using `wget` and loaded into a pandas DataFrame.
- The dataset contains medical questions and answers, which are essential for fine-tuning the model.

#### 4. Data Preprocessing

- **Handling Missing Values:**
  - Rows with missing 'Question' or 'Answer' fields are dropped to ensure data quality.
- **Removing Duplicates:**
  - Duplicate entries based on the 'Question' and 'Answer' columns are removed.
- **Focus Categories Extraction:**
  - The total number of categories in the 'Focus' column is calculated.
  - The top 100 focus categories are identified and their names are extracted for further processing.

#### 5. Creating Training and Validation Sets

- Data is split into training and validation sets per focus category.
- **Sample Reduction:**
  - The number of samples per category is reduced to speed up training, which is practical given resource constraints.
  - For categories with more data, only a subset is used.
- **Shuffling Data:**
  - Data is shuffled to ensure a random distribution of samples.
- **Training and Validation Lists:**
  - Separate lists for training and validation data are created and concatenated into final DataFrames.
- **Resulting Data Samples:**
  - The number of training and validation samples is printed for verification.

#### 6. Preparing Data for Model Input

- **Combining Questions and Answers:**
  - Questions and answers are combined with special tokens to create a suitable input format for the model.
  - Special tokens used are `<question>`, `<answer>`, and `<end>`.
- **Saving to Text Files:**
  - The combined data is saved into `train_data.txt` and `val_data.txt` for both training and validation.

#### 7. Tokenizer and Model Setup

- **Loading the GPT-2 Tokenizer:**
  - The GPT-2 tokenizer is loaded using `GPT2Tokenizer.from_pretrained('gpt2')`.
- **Adding Special Tokens:**
  - Special tokens are added to the tokenizer's vocabulary using `tokenizer.add_special_tokens()`.
- **Setting Padding Token:**
  - The `pad_token` is set to the `eos_token` to handle padding.
- **Loading the GPT-2 Model:**
  - The GPT-2 model is loaded using `GPT2LMHeadModel.from_pretrained('gpt2')`.
- **Resizing Model Embeddings:**
  - The model's embeddings are resized to match the updated tokenizer vocabulary size using `model.resize_token_embeddings(len(tokenizer))`.
- **Enabling Gradient Checkpointing:**
  - Gradient checkpointing is enabled using `model.gradient_checkpointing_enable()` to reduce memory usage during training.

#### 8. Tokenizing the Data

- **Loading Datasets:**
  - The datasets are loaded from the text files using `load_dataset('text', data_files=...)`.
- **Tokenization Function:**
  - A tokenization function is defined to tokenize the text data.
  - Parameters:
    - `truncation=True`
    - `max_length=256` to manage memory usage and training time.
- **Applying Tokenization:**
  - The tokenization function is applied to the datasets using the `map` method.

#### 9. Data Collator Creation

- A data collator is created for language modeling using `DataCollatorForLanguageModeling`.
- Parameters:
  - `tokenizer=tokenizer`
  - `mlm=False` (since we're doing language modeling, not masked language modeling)
  - `return_tensors='pt'`
- This helps in dynamically padding inputs during batching and prepares data for the language modeling task.

#### 10. Fine-Tuning the GPT-2 Model

- **Setting Training Arguments:**
  - Training arguments are configured using `TrainingArguments`.
- **Configuring Early Stopping:**
  - Early stopping is configured with `EarlyStoppingCallback` to prevent overfitting.
- **Training the Model:**
  - The model is trained using the `Trainer` class.
  - The training process includes evaluation at each epoch and early stopping if no improvement is observed.
- **Saving the Model and Tokenizer:**
  - After training, the model and tokenizer are saved to the specified output path.
- **Verifying Saved Files:**
  - The contents of the model directory are listed to verify that all necessary files are saved, including `pytorch_model.bin`.

#### 11. Testing the Fine-Tuned Model

- **Loading the Fine-Tuned Model and Tokenizer:**
  - The fine-tuned model and tokenizer are loaded from the saved directory.
- **Defining the Response Generation Function:**
  - A `generate_response` function is defined to generate responses to prompts.
  - The function:
    - Encodes the prompt using the tokenizer.
    - Generates the output sequence using `model.generate()`.
    - Decodes the response using the tokenizer.
- **Testing with Sample Prompts:**
  - The model is tested with sample prompts related to medical queries, such as:
    - "What are the symptoms of diabetes?"
    - "How can I lower my blood pressure?"
  - Outputs are printed to assess the model's performance.

#### 12. Comparing with the Original GPT-2 Model

- **Loading the Original GPT-2 Model:**
  - The original GPT-2 model and tokenizer are loaded without fine-tuning.
- **Testing Both Models:**
  - Both the fine-tuned and original models are tested with the same prompts.
- **Evaluating Improvements:**
  - The responses are compared to highlight the improvements gained from fine-tuning on the MedQuAD dataset.

### Files
[Fine-Tuning GPT-2 on MedQuAD for Medical Question Answering.ipynb](Fine-Tuning%20GPT-2%20on%20MedQuAD%20for%20Medical%20Question%20Answering.ipynb): The main Colab notebook containing the fine-tuning code and workflow.

### Business Applications

A fine-tuned model such as this can be used in healthcare to:

- Provide automated responses to medical queries, offering immediate assistance to patients.
- Improve patient care by making medical information more accessible.
- Reduce the workload for medical professionals by handling routine questions.
- Enhance telemedicine services by integrating AI-driven Q&A capabilities.
