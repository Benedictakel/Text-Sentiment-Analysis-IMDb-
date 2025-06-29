# 🎭 Text Sentiment Analysis (IMDb)

This project implements a **sentiment analysis model** to classify movie reviews from the IMDb dataset as **positive or negative**, using deep learning-based Natural Language Processing (NLP) techniques.



## 📑 Table of Contents

* [Introduction](#introduction)
* [Dataset](#dataset)
* [Technologies Used](#technologies-used)
* [Installation](#installation)
* [Usage](#usage)
* [Model Architecture](#model-architecture)
* [Project Structure](#project-structure)
* [Results](#results)
* [Contributing](#contributing)
* [License](#license)
* [Contact](#contact)



## 📝 Introduction

**Sentiment Analysis** is a common NLP task used to determine the **emotional tone** behind a body of text. In this project, we build a **deep learning model** to classify IMDb movie reviews as:

✅ **Positive (1)**
✅ **Negative (0)**

The model can be used by streaming platforms, review aggregators, and social media analytics tools to understand customer opinions at scale.



## 📚 Dataset

* **Dataset:** [IMDb Movie Reviews]()
* **Size:** 50,000 reviews (25,000 for training, 25,000 for testing)
* **Labels:**

  * 0: Negative
  * 1: Positive



## ✨ Features

✔️ Load and preprocess IMDb dataset

✔️ Tokenization and padding for text sequences

✔️ Build LSTM-based or Bidirectional LSTM model using Keras

✔️ Train and evaluate model performance

✔️ Predict sentiment of custom input reviews



## 🛠️ Technologies Used

* **Python 3**
* **TensorFlow / Keras**
* `numpy`
* `pandas`
* `matplotlib`
* **Jupyter Notebook**



## ⚙️ Installation

1. **Clone the repository**

```bash
git clone https://github.com/yourusername/Text-Sentiment-Analysis-IMDb.git
cd Text-Sentiment-Analysis-IMDb
```

2. **Create and activate a virtual environment (optional)**

```bash
python -m venv venv
source venv/bin/activate  # For Linux/Mac
venv\Scripts\activate     # For Windows
```

3. **Install dependencies**

```bash
pip install -r requirements.txt
```

4. **Launch Jupyter Notebook**

```bash
jupyter notebook
```



## ▶️ Usage

1. Open `IMDb_Sentiment_Analysis.ipynb` in Jupyter Notebook.
2. Run cells sequentially to:

   * Load and preprocess the dataset
   * Tokenize and pad text sequences
   * Build and compile the sentiment analysis model
   * Train the model with validation
   * Evaluate the model on test data
   * Predict sentiment of custom text inputs



## 🏗️ Model Architecture

Sample **LSTM-based model architecture**:

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Bidirectional

model = Sequential([
    Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=max_len),
    Bidirectional(LSTM(64)),
    Dense(1, activation='sigmoid')
])

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
```



## 📁 Project Structure

```
Text-Sentiment-Analysis-IMDb/
 ┣ IMDb_Sentiment_Analysis.ipynb
 ┣ requirements.txt
 ┗ README.md
```



## 📈 Results

* **Training Accuracy:** *e.g. 95%*
* **Test Accuracy:** *e.g. 89%*

Sample prediction:

```
Review: "This movie was absolutely fantastic, I loved every moment!"
Predicted Sentiment: Positive
```



## 📊 Example Prediction

```python
review = ["The plot was boring and acting was terrible."]
sequence = tokenizer.texts_to_sequences(review)
padded = pad_sequences(sequence, maxlen=max_len)
prediction = model.predict(padded)
sentiment = "Positive" if prediction >= 0.5 else "Negative"
print("Predicted Sentiment:", sentiment)
```



## 🤝 Contributing

Contributions are welcome to:

* Experiment with GRU or Transformer-based models for better accuracy
* Integrate pre-trained embeddings (e.g., GloVe, Word2Vec)
* Deploy as a REST API using Flask or FastAPI for real-time sentiment analysis

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/YourFeature`)
3. Commit your changes (`git commit -m 'Add YourFeature'`)
4. Push to the branch (`git push origin feature/YourFeature`)
5. Open a pull request



## 📄 License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.



## 📬 Contact

**Ugama Benedicta Kelechi**
[LinkedIn](www.linkedin.com/in/ugama-benedicta-kelechi-codergirl-103041300) | [Email](mailto:ugamakelechi501@gmail.com) | [Portfolio](#)



### ⭐️ If you find this project useful, please give it a star!

