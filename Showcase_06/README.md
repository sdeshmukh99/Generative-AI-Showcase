## Overview

This project focuses on building a **Spam Detection Model** using machine learning techniques to automatically classify messages as "spam" or "ham" (non-spam). By leveraging neural networks, this model aims to provide an efficient solution for filtering unwanted or malicious messages, enhancing user experience and security.

**Learning Objectives:**
- **Data Acquisition & Preprocessing:** Understand how to gather, clean, and prepare textual data for machine learning tasks.
- **Model Building:** Learn to design and implement neural network architectures tailored for text classification.
- **Training & Evaluation:** Gain insights into training models effectively and evaluating their performance using relevant metrics.
- **Visualization:** Explore techniques to visualize learned embeddings, offering a deeper understanding of how the model interprets text data.

**Dataset:**
The project utilizes the `spam.csv` dataset, which comprises a collection of messages labeled as either "spam" or "ham." This dataset serves as the foundation for training and evaluating the spam detection model, ensuring it can distinguish between unwanted and legitimate messages accurately.

## Code Walkthrough

### 1. Get the Dataset and Import Necessary Packages

**Purpose:**  
This initial section sets up the environment by downloading the required dataset and importing all necessary Python libraries essential for data manipulation, visualization, and building the machine learning model.

**Key Actions:**
- **Dataset Download:**  
  Utilizes a command to fetch the `spam.csv` file from a specified URL, ensuring the data is available locally for processing.
  
- **Library Imports:**  
  Imports essential libraries:
  - **Pandas & NumPy:** For efficient data manipulation and numerical operations.
  - **Matplotlib:** For creating visualizations to interpret data and model performance.
  - **TensorFlow:** For constructing and training the neural network model.
  - **Scikit-learn:** For splitting the dataset into training and testing sets and encoding categorical labels.

### 2. Read the Data

**Purpose:**  
This section focuses on loading the dataset into a workable format and performing initial data cleaning to prepare it for model training.

**Key Actions:**
- **Loading the Dataset:**  
  Reads the `spam.csv` file into a Pandas DataFrame, specifying the correct encoding to handle any special characters.
  
- **Data Cleaning:**  
  Removes unnecessary columns that do not contribute to the analysis and renames existing columns for clarity:
  - **Dropping Columns:** Eliminates columns named 'Unnamed: 2', 'Unnamed: 3', and 'Unnamed: 4'.
  - **Renaming Columns:** Changes 'v1' to 'label' and 'v2' to 'text' for better readability and understanding.
  
- **Data Exploration:**  
  Prints the first few rows of the dataset and provides a summary of its structure to verify successful loading and cleaning.
  
- **Class Distribution:**  
  Analyzes the balance between spam and ham messages to identify any class imbalance that might affect model training.
  
- **Label Encoding:**  
  Transforms categorical labels ('spam' and 'ham') into numerical values using Label Encoding, making them suitable for machine learning algorithms.
  
- **Data Splitting:**  
  Divides the dataset into training and testing sets, allocating 80% for training and 20% for testing to evaluate model performance on unseen data.

### 3. Apply the Vectorization

**Purpose:**  
Textual data must be converted into numerical representations before being processed by neural networks. This section handles the transformation of text into numerical vectors and sets up the neural network architecture.

#### 3.1 Define and Compile the Model

**Key Actions:**
- **Model Parameters:**  
  Sets crucial parameters for the model:
  - **Embedding Dimension (`embedding_dim`):** Determines the size of the word vectors.
  - **Vocabulary Size (`max_vocab_size`):** Limits the number of unique words considered in the vocabulary.
  - **Sequence Length (`max_sequence_length`):** Specifies the maximum length of input sequences (messages) in terms of word count.
  
- **Model Architecture:**  
  Constructs a Sequential Neural Network with the following layers:
  - **Input Layer:** Defines the shape of the input data.
  - **Embedding Layer:** Converts word indices into dense vectors of fixed size, capturing semantic relationships.
  - **Global Average Pooling Layer:** Reduces the dimensionality by averaging the embeddings, simplifying the input for the subsequent layers.
  - **Dense Layer with ReLU Activation:** Introduces non-linearity, enabling the model to learn complex patterns.
  - **Output Layer with Sigmoid Activation:** Outputs a probability indicating whether a message is spam or ham.
  
- **Model Compilation:**  
  Configures the learning process by specifying:
  - **Loss Function:** Uses binary cross-entropy, suitable for binary classification tasks.
  - **Optimizer:** Employs the Adam optimizer for efficient gradient descent.
  - **Metrics:** Tracks accuracy to evaluate model performance during training.

#### 3.2 Create Text Vectorization Layer and Prepare Dataset

**Key Actions:**
- **Text Vectorization:**  
  Implements a `TextVectorization` layer to convert textual messages into sequences of integers based on the vocabulary.
  
- **Adaptation:**  
  Fits the vectorizer to the training data to build the vocabulary and learn word frequency.
  
- **Dataset Preparation:**  
  Converts the training and testing data into TensorFlow datasets, applying batching for efficient processing.
  
- **Vectorization Function:**  
  Defines a function to apply the vectorization to each message, transforming text into numerical sequences.
  
- **Optimization:**  
  Applies caching and prefetching to the datasets to enhance training performance by preparing data in advance.

### 4. Extract the Embeddings before Training

**Purpose:**  
Before training the model, it's insightful to examine the initial state of the embeddings. This section initializes the model's weights and retrieves the initial embeddings from the Embedding layer.

**Key Actions:**
- **Model Initialization:**  
  Passes a dummy input through the model to initialize the weights, ensuring that the model's layers are properly built.
  
- **Layer Verification:**  
  Prints out the number of layers and their names to confirm the correct architecture.
  
- **Embedding Extraction:**  
  Accesses the Embedding layer by its assigned name and retrieves the initial embedding weights, which are matrices representing word vectors before any training has occurred.
  
- **Shape Confirmation:**  
  Prints the shape of the initial embeddings to verify that they match the expected dimensions based on the vocabulary size and embedding dimension.

### 5. Train the Model

**Purpose:**  
This section is dedicated to training the neural network using the prepared datasets. It involves fitting the model to the training data and validating its performance on the testing set.

**Key Actions:**
- **Model Training:**  
  Initiates the training process for a specified number of epochs (iterations), during which the model learns to distinguish between spam and ham messages.
  
- **History Tracking:**  
  Stores the training history, which includes metrics like loss and accuracy for both training and validation sets, to monitor the model's learning progress.

### 6. During Training Collect and Plot Metrics

**Purpose:**  
Visualizing training metrics helps in understanding how well the model is learning and whether it's generalizing effectively to unseen data. This section plots the loss and accuracy over epochs for both training and validation sets.

**Key Actions:**
- **Loss Plotting:**  
  Creates a subplot showing the decrease in training and validation loss over each epoch, indicating the model's ability to minimize errors.
  
- **Accuracy Plotting:**  
  Generates a subplot displaying the increase in training and validation accuracy over epochs, reflecting the model's improving ability to correctly classify messages.
  
- **Visualization Layout:**  
  Adjusts the layout to ensure plots are neatly arranged and labels are clear, enhancing readability and interpretability.

### 7. Calculate and Display the Test Performance

**Purpose:**  
After training, it's essential to evaluate the model's performance on the testing set to assess its ability to generalize to new, unseen data. This section computes and displays the final accuracy and loss on the test dataset.

**Key Actions:**
- **Model Evaluation:**  
  Runs the model on the test dataset to obtain loss and accuracy metrics, providing a quantitative measure of performance.
  
- **Performance Reporting:**  
  Prints the test accuracy and loss with high precision, offering clear insights into how well the model performs in real-world scenarios.

### 8. Extract the Embeddings after Training

**Purpose:**  
Post-training, examining the learned embeddings can reveal how the model has captured semantic relationships between words. This section retrieves the trained embeddings from the Embedding layer.

**Key Actions:**
- **Embedding Retrieval:**  
  Accesses the Embedding layer by name and extracts the updated embedding weights, which now reflect the model's learned representations.
  
- **Shape Confirmation:**  
  Prints the shape of the learned embeddings to ensure consistency with the initial embeddings, verifying successful extraction.

### 9. Reduce the Embeddings using PCA

**Purpose:**  
High-dimensional embeddings are challenging to visualize. This section applies Principal Component Analysis (PCA) to reduce the embeddings' dimensionality, making it feasible to plot them in a 2D space.

**Key Actions:**
- **PCA Implementation:**  
  Utilizes the PCA technique from Scikit-learn to compress the 256-dimensional embeddings into 2 principal components.
  
- **Dimensionality Reduction:**  
  Transforms the learned embeddings into a 2D space, retaining as much variance as possible to preserve meaningful relationships between words.
  
- **Shape Confirmation:**  
  Prints the shape of the reduced embeddings to confirm successful dimensionality reduction, resulting in a (10000, 2) matrix.

### 10. Plot the Embeddings in 2D

**Purpose:**  
Visualizing the reduced embeddings provides an intuitive understanding of how the model perceives the relationships between different words. This section plots the embeddings, highlighting specific words of interest.

**Key Actions:**
- **Scatter Plot Creation:**  
  Generates a 2D scatter plot of the reduced embeddings, allowing for the visualization of word distributions and clusters.
  
- **Word Annotation:**  
  Selects specific words (e.g., 'free', 'win', 'call') and annotates their positions on the plot to observe their relationships and proximities to other words.
  
- **Layout Adjustment:**  
  Ensures that the plot is neatly arranged with clear labels and titles, enhancing readability and interpretability of the visualized data.

## Conclusion

This project successfully demonstrates the process of building a neural network-based spam detection model, from data acquisition and preprocessing to model training, evaluation, and insightful visualization of learned embeddings.

