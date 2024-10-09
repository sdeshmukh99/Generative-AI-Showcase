(!) - key sections

## Decoder-Only Transformer (GPT) from Scratch

### Objective

This project aims to build a decoder-only Transformer architecture, similar to GPT, from scratch. It focuses on understanding and implementing critical components such as causal self-attention, positional encoding, and the Transformer block. We further use the model to generate text based on training using a real-world sentiment analysis dataset. The goal is to demonstrate the workings of a Generative Pre-trained Transformer (GPT) and apply it for text generation using PyTorch.

### Overview

The implementation is organized into several sections, starting from installing the required packages to training a custom GPT-like model for text generation. Each section aims to explain a core component of the architecture in a modular, understandable way. This README file walks through each section and details the functionality implemented in every part of the code.


### 0. Package Installation and Imports
To start, we install necessary libraries like transformers and datasets. These libraries are essential for handling pre-trained models and datasets. We import relevant Python packages for deep learning (torch), data manipulation (numpy), plotting (matplotlib), and handling datasets (datasets, transformers). These imports allow us to build, train, and visualize the model as well as prepare our dataset.

### 1. Causal Self-Attention and Transformer Components

#### 1.1. Causal Self-Attention Class (!)

(This is the core mechanism for attention, allowing the model to focus on different parts of the input sequence. It is fundamental to understanding how the model builds context when generating text.)

The CausalSelfAttention class implements the core attention mechanism used in Transformer models. It allows the model to focus on different parts of the input sequence dynamically, which is key to generating context-aware text.

- The class initializes key, query, and value matrices, which are used to calculate attention scores.

- A causal mask is also created to ensure that the model cannot "see" future tokens when generating text. This is critical for generating sequences coherently, one token at a time.

- The forward method applies the linear transformations and computes the attention weights, eventually returning the weighted sum of values.

#### 1.2. Transformer Block Class (!)

(This block integrates the attention mechanism with normalization and feed-forward layers, showing how the architecture layers on more complex features for each token.)

The TransformerBlock class builds on top of the CausalSelfAttention to add normalization and a feed-forward network.

- The block consists of layer normalization, multi-head attention, and a fully connected feed-forward network.

- Dropout is used to prevent overfitting.

- The block is designed to add learned transformations to each token representation, improving its understanding of the context.

#### 1.3. Positional Encoding Class (!)

(This is essential for giving the model a sense of order in the input sequence, which is crucial for tasks involving natural language.)

The PositionalEncoding class helps the model understand the position of each token in the input sequence.

- Since Transformers do not have inherent knowledge of word order, positional encoding provides a way for the model to represent the sequential nature of the data.

- It generates sine and cosine functions that are added to the input embeddings, allowing the model to learn relative positions.

### 2. Decoder-Only Transformer (GPT) Architecture

#### 2.1. Decoder Class (!)

(Represents the full GPT-like model, combining embedding, positional encoding, and multiple Transformer blocks. This is the architecture that brings all components together.)

The Decoder class represents the GPT architecture, which consists of multiple Transformer blocks stacked together.

- The model begins with an embedding layer, converting token indices to vector representations.

- Positional encoding is added to these embeddings to provide sequence information.

- A sequence of Transformer blocks is applied, each contributing more sophisticated transformations to the input.

- Finally, a linear layer projects the transformed embeddings back to vocabulary size, allowing us to generate output tokens.

#### 2.2. Testing the Decoder with Dummy Data

Before training, the model is tested using dummy data to ensure it works as expected.

- Random input sequences are passed through the model, with and without padding, to verify the output shape and functionality.

- This step is crucial for debugging and validating the architecture.

### 3. Training the Decoder as a Causal Language Model

#### 3.1. Load and Tokenize the Data

We use the glue dataset, specifically the sst2 task, which provides a dataset of sentiment-labeled sentences from movie reviews. The dataset contains sentences with binary labels indicating whether the sentiment is positive or negative, making it ideal for training models on sentiment analysis.

- A pre-trained tokenizer (distilbert-base-cased) is used to convert text data into token IDs.

- The dataset is tokenized and prepared for training, with padding handled using the DataCollatorWithPadding.

#### 3.2. Training Process (!)

(This section is crucial for understanding how the model is trained to learn from input sequences and generate coherent text. It involves the loss calculation and optimization steps.)

The training loop is defined in the train function.

- Cross-entropy loss is used, ignoring the padding tokens, to train the model.

- The model processes batches of input tokens, learns from the error, and updates its weights accordingly.

- The target tokens are created by shifting the input sequence, allowing the model to learn the next token in sequence.

The training process runs for four epochs, with losses printed at the end of each epoch to track progress.

### 4. Validation and Text Generation

#### 4.1. Validation Process

Once the model is trained, we validate it using a small validation set.

- The model is put in evaluation mode, and predictions are generated for a batch of validation data.

- This allows us to verify if the model has learned useful patterns from the training data.

#### 4.2. Sample Validation Output

We decode the generated tokens from the model output to check if the predictions make sense.

The predicted tokens are compared to the input, and their decoded forms are displayed for easier understanding.

This step helps in visualizing how well the model is performing.

#### 4.3. Extracting the Next Word Prediction

- The next word is predicted by finding the token with the highest probability for the current position.

- This involves extracting the probabilities from the model output and decoding the most likely next token.

### 5. Generating Text Using the Model (!)

(This showcases the end goal of the assignment—using the trained model to generate meaningful text. It demonstrates the practical application of all preceding steps.)

Finally, we generate new text using the trained model. For instance, the model generated the output: '[CLS] it's a good deal of the film [SEP]' after training. This output indicates that the model successfully learned to generate coherent text based on the given prompt, demonstrating its understanding of the language structure.

- A prompt ("it's a") is tokenized and fed into the model.

- The model then generates new tokens iteratively until a stopping condition is met (e.g., end-of-sequence token or a predefined length).

- The generated text is decoded to produce readable output, showcasing the generative ability of the trained model.

### Summary

This assignment demonstrates the implementation of a decoder-only Transformer from scratch, similar to the architecture used in GPT. It covers key concepts like causal self-attention, Transformer blocks, positional encoding, and built a text generator model. The model was trained on sentiment data and was used for text generation, providing insights into the underlying mechanisms of modern NLP models like GPT.

This code not only helped me understand how GPT models function internally but also offered hands-on experience with deep learning frameworks, tokenization, and training complex neural networks. It is helpful to build and understand large-scale NLP models, which is a crucial skill in AI product management and related fields.
