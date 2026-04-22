📘 Next Word Prediction using RNN, LSTM & GRU

📌 Project Overview

This project focuses on building a Next Word Prediction system using deep learning techniques such as Recurrent Neural Networks (RNN), Long Short-Term Memory (LSTM), and Gated Recurrent Units (GRU).

The model is trained on the Shakespeare Hamlet dataset from the NLTK Gutenberg Corpus to learn language patterns and predict the next word in a sequence.

🎯 Objective

The main objectives of this project are:

To understand sequence modeling in NLP
To implement deep learning-based language models
To explore and compare RNN, LSTM, and GRU architectures
To generate meaningful next-word predictions based on input text

📂 Dataset
Source: NLTK Gutenberg Corpus
Text: Shakespeare – Hamlet
Why this dataset?
Rich in vocabulary and linguistic patterns
Suitable for learning sequential dependencies
Common benchmark for language modeling tasks

⚙️ Project Workflow
1. Data Collection & Understanding
Loaded dataset using NLTK
Analyzed:
Total number of words
Sample text

2. Text Preprocessing
Converted text to lowercase
Removed unnecessary characters
Tokenized text using Keras Tokenizer
Generated word index dictionary

3. Sequence Generation
Created input sequences using n-gram approach

Example:

"I am learning deep learning"
→ ["I am", "I am learning", "I am learning deep", ...]
Converted sequences into numerical format

4. Padding & Feature Engineering
Applied padding using pad_sequences
Split data into:
Predictors (X)
Labels (y)
Converted labels into categorical format

5. Model Design
Implemented deep learning models using:
Embedding Layer
LSTM / RNN / GRU Layer(s)
Dense Layer with Softmax activation
Compilation:
Loss Function: categorical_crossentropy
Optimizer: Adam

6. Model Training & Evaluation
Trained the model on prepared sequences
Visualized:
Training loss vs epochs
Evaluated model performance
Analyzed:
Overfitting / Underfitting

7. Next Word Prediction
Developed a prediction function:
Input: Seed text
Output: Predicted next word
Tested model on multiple input sequences

8. Analysis & Improvements
Limitations of LSTM:
Computationally expensive
Struggles with very long sequences
Limited parallelization
Possible Improvements:
Bidirectional LSTM
Transformer-based models
Larger datasets
Hyperparameter tuning
LSTM vs Transformer:
Feature	LSTM	Transformer
Architecture	Sequential	Parallel
Speed	Slower	Faster
Long Dependencies	Moderate	Excellent
Performance	Good	State-of-the-art

📊 Results
Successfully trained a language model capable of predicting next words
Observed learning trends through loss graphs
Generated coherent predictions for test inputs

Bidirectional LSTM
Comparison between RNN, LSTM, and GRU

🛠️ Tech Stack
Python
TensorFlow / Keras
NLTK
NumPy / Pandas
Matplotlib

📁 Project Structure
├── notebook.ipynb
├── README.md
└── report.pdf
📌 Submission Details

As per assignment requirements :

Jupyter Notebook (.ipynb)
Summary Report including:
Methodology
Results
Graphs
Observations

👨‍💻 Author
[Rahul kumar]
MCA – Semester 2
Course: IMDAI (CSET-654)