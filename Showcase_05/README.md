## Intent Classification Using BERT

### Overview

This project demonstrates intent classification using a pre-trained BERT model, treating intent classification as a text classification problem. The dataset includes questions, responses, and intents. We use DistilBERT, a lightweight version of BERT, because it is highly effective for natural language understanding tasks, especially for capturing context and meaning in text, while being faster and more efficient for smaller datasets. The workflow involves acquiring the dataset, processing it, loading a pre-trained BERT model, fine-tuning it, and writing functions for intent prediction and response retrieval. Finally, we integrate these functions to validate the complete workflow.

This project uses Python in Google Colab and leverages Hugging Face's Transformers library for implementation. The goal is to understand the entire pipeline: from data preparation and tokenization to training and inference, while also practicing model fine-tuning and evaluation.

### Code Walkthrough
The project is divided into multiple sections, each accomplishing specific tasks for the overall objective of intent classification.

#### 1. Setup Steps

##### 1.1 Install Required Libraries
We need some specific Python libraries, such as `transformers` and `datasets`, which we install using the following command:
These libraries provide the necessary tools for working with transformer models and datasets.

##### 1.2 Download Dataset
The dataset (`Intent.json`) used here is downloaded directly using a wget command:
This JSON file contains intents and the corresponding questions that map to those intents.

##### 1.3 Import Necessary Packages
We import all the required packages, including NumPy, Pandas, JSON, PyTorch, and transformers from Hugging Face. These libraries help in data manipulation, model creation, and training.

##### 1.4 Check for GPU Availability
We check if a GPU is available and set the device accordingly:
This helps speed up training by using the GPU if available.

#### 2. Data Preparation Steps
##### 2.1 Load Dataset from JSON File
We load the dataset from the downloaded JSON file into a variable `data`.

##### 2.2 Exploratory Data Analysis
We take a quick look at each entry in the dataset to understand its structure.
This helps us understand the different intents and the format of the data.

##### 2.3 Convert JSON Data to DataFrame and Create a Copy
We convert the JSON structure into a Pandas DataFrame for easier manipulation and also create a copy (`raw_dataset_base`) for reference.

##### 2.4 Drop Unnecessary Columns and Display Dataset
We drop columns that are not needed for the intent classification task to keep only the useful features.
This step simplifies the dataset, keeping only relevant columns.

##### 2.5 Explode the 'text' Column
The 'text' column contains multiple questions for each intent, stored as a list. We "explode" the column to create a new row for each question.
This increases the dataset size, giving the model more training samples, which improves its performance.

##### 2.6 Encode Intent Labels to Numeric Values
We encode the intent labels into numerical values using `LabelEncoder`.
This transformation is required because machine learning models work better with numerical values.

##### 2.7 Split the Data into Training, Validation, and Test Sets
We split the data into training, validation, and test sets. The training set is used to train the model, the validation set to tune hyperparameters, and the test set to evaluate the model’s final performance.

##### 2.8 Convert DataFrames to Dataset Format and Display Lengths
We convert the DataFrames to Hugging Face's `Dataset` format for compatibility with the transformer models.
This step ensures the data is ready for training with transformers.

#### 3. Data Tokenization Steps
##### 3.1 Load Pre-trained BERT Tokenizer
We load the tokenizer for the BERT model (`distilbert-base-cased`).
The tokenizer is responsible for converting text to a format that the model can understand.

##### 3.2 Define Tokenization Function
We define a function to tokenize each sample in the dataset.
This function tokenizes each question, ensuring that they all have the same length.

##### 3.3 Tokenize Train, Validation, and Test Datasets
We apply the tokenization function to each dataset.

##### 3.4 Prepare the Tokenized Datasets for Training
We remove unnecessary columns and format the data for PyTorch compatibility.
We then apply this function to the tokenized datasets.

#### 4. Model Loading and Training Steps
##### 4.1 Load a Pre-Trained BERT Model
We load the pre-trained BERT model with a classification head to fine-tune it for intent classification.

##### 4.2 Define Performance Metrics for Evaluation
We define metrics (accuracy and F1 score) to evaluate the model’s performance.

##### 4.3 Set Training Parameters
We configure training parameters, including learning rate, batch size, number of epochs, and early stopping.
We also use early stopping to prevent overfitting if the model stops improving for a few epochs.

##### 4.4 Train the Model Using the Trainer API
We create a `Trainer` instance to handle training, validation, and logging.
The early stopping callback will stop training if no improvements are observed for 5 consecutive epochs.

#### 5. Prediction and Response Steps
##### 5.1 Define Function to Predict Intent for a Given Question
We define a function to predict the intent of a given user query.
This function tokenizes the sentence, passes it through the model, and returns the predicted intent.

##### 5.2 Test the Prediction Function
We test the function with a sample question.

##### 5.3 Define Function to Get a Response Based on Intent
We define a function to retrieve a response based on the predicted intent.
This function uses the predicted intent to find a corresponding response.

##### 5.4 Test the Response Function
We test the response function.

##### 5.5 Define Function to Predict Intent and Provide a Response
We define a function that combines intent prediction and response retrieval.
This function takes a user query, predicts the intent, and retrieves the appropriate response.

##### 5.6 Test the Integrated Prediction and Response Function
We test the complete prediction and response function.
This integrated function is a demonstration of the complete workflow from user query to final response.

### Summary
This project showcases the full pipeline of preparing, training, and deploying a BERT-based intent classification model. We started by preparing the dataset, followed by tokenization and model training. Finally, we tested the model's ability to predict intents and provide appropriate responses. 
