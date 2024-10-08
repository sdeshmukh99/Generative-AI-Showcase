### Sentiment Analysis Using Custom Transformer Encoder

#### **1. Overview of the Assignment**
In this assignment, I implemented a custom **transformer encoder** architecture and used it for sentiment analysis on text data, specifically the IMDB and SST-2 (Stanford Sentiment Treebank) datasets. The aim was to design and train a model to classify text-based inputs (such as movie reviews) as positive or negative.

**Transformer Encoder in the Bigger Picture:**
The transformer architecture allows models to efficiently process entire sequences of text by utilizing mechanisms like **self-attention**. In this project, I designed the encoder portion of the transformer model, which is responsible for capturing contextual relationships in text, a key element for tasks like sentiment analysis, text generation, and machine translation.

---

#### **2. Importing Packages**
The first section of the code imports essential libraries, including:
- **PyTorch**: For building and training neural networks.
- **Hugging Face Transformers**: Provides pre-trained models and tokenizers.
- **Pandas**, **Numpy**, and **Matplotlib**: For handling and visualizing data.

These libraries allow us to construct models, handle text tokenization, manage datasets, and visualize training progress.

---

#### **3. Loading the Dataset**
I used the **IMDB** dataset for sentiment analysis and loaded it using the Hugging Face Datasets library. This dataset contains movie reviews labeled as either positive or negative. The **IMDB** dataset was explored to understand the data structure before building the transformer. I also used the SST-2 dataset from the GLUE benchmark. The SST-2 dataset provides an excellent test of binary sentiment classification and is widely used in natural language processing tasks.

---

#### **4. Designing the Transformer Encoder**
At the core of this project is the transformer encoder, which processes input text sequences and generates contextual embeddings.

- **Multi-Head Attention**: This allows the model to focus on multiple aspects of the input at once, giving it the ability to capture complex relationships between words.
- **Positional Encoding**: Since transformers are inherently sequence-agnostic, I added positional encodings to inject the order of tokens into the model, ensuring that the model knows where each word appears in the sequence.

The encoder was built using PyTorch, with a multi-head attention mechanism, a feed-forward network, and layer normalization.

---

#### **5. Forward Pass and Training the Model**
To test the encoder, I designed a forward pass, where input sequences are tokenized, passed through the transformer blocks, and processed to produce predictions. The model’s goal was to predict whether a movie review was positive or negative.

During training:
- **Loss Function**: Cross-entropy loss was used, which is ideal for classification tasks.
- **Optimization**: The model was trained using the Adam optimizer, which adjusts learning rates during training to ensure efficient convergence.

The model was trained over several epochs, adjusting its weights based on the training data, with validation accuracy being monitored throughout.

---

#### **6. Evaluation of the Model**
Once the model was trained, it was evaluated using accuracy metrics on both the training and validation datasets. The model's ability to generalize was measured by its performance on unseen validation data from both the IMDB and SST-2 datasets, allowing for a robust evaluation of its performance. This step demonstrated how well the custom-built transformer encoder could learn the task of sentiment classification. 

---

#### **7. Testing on Custom Sentences**
I also provided a mechanism for evaluating the model on custom sentences. This step allowed for real-world testing of the model, where I input various sentences (e.g., "I love this movie!") to see how well the model predicted sentiment. This section highlights the practical applicability of the model beyond standard datasets.

---

#### **8. Visualizing Model Outputs and Understanding Internal Representations**
Finally, I used **Principal Component Analysis (PCA)** to visualize the learned embeddings from the model's CLS token (a representation of the entire input sequence). This helps us inspect how the model learned to represent different sentences in a lower-dimensional space. Additionally, the internal weights and embeddings were examined to understand what the model learned during training.

---

### Conclusion
This showcase illustrates the power of transformers for text classification. By designing a custom transformer encoder, I showcased the ability to train a model from scratch, optimize its performance, and interpret its outputs. This process not only deepened my understanding of transformer architectures but also provided practical experience in implementing state-of-the-art NLP models.
