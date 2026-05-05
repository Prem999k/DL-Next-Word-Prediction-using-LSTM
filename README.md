# 🧠 DL Word & Sentence Completion using LSTM

## 📌 Project Overview

This project is a **Deep Learning-based Text Generation System** that predicts the **next word** and can **automatically complete sentences** using an **LSTM (Long Short-Term Memory) model**.

The model learns patterns from text data and generates context-aware predictions based on user input.
https://dl-next-word-prediction-using-lstm-premkumark.streamlit.app/
---

## 🚀 Features

* 🔤 Next Word Prediction
* 📝 Automatic Sentence Completion
* ⚡ Real-time predictions using Streamlit
* 🧠 LSTM-based Deep Learning model
* 🎯 Context-aware text generation

---

## 🛠️ Tech Stack

* **Python**
* **TensorFlow / Keras**
* **NumPy**
* **Streamlit**

---

## 📂 Project Structure

```
├── app.py                # Streamlit application
├── requirements.txt     # Dependencies
├── runtime.txt          # Python version (for deployment)
├── lstm_model.h5        # Trained LSTM model
├── tokenizer.pkl        # Tokenizer object
├── max_len.pkl          # Maximum sequence length
```

---

## ⚙️ How It Works

1. User enters a sentence
2. Text is converted into sequences using a tokenizer
3. LSTM model predicts the next word
4. For sentence completion:

   * Words are generated iteratively
   * Stops when sentence ends or max length reached

---

## ▶️ Run Locally

```bash
# Clone repository
git clone https://github.com/your-username/DL-Next-Word-Prediction-LSTM.git

# Navigate to project folder
cd DL-Next-Word-Prediction-LSTM

# Install dependencies
pip install -r requirements.txt

# Run Streamlit app
streamlit run app.py
```

---

## 🌐 Deployment

This app can be deployed on **Streamlit Cloud**.

👉 Make sure:

* `requirements.txt` is correct
* `runtime.txt` contains `python-3.10`

---

## 🎯 Example

**Input:**

```
life is
```

**Output:**

```
life is beautiful when you believe in yourself and keep moving forward
```

---

## 📈 Future Improvements

* 🔥 Improve prediction accuracy
* 🔍 Add beam search for better text generation
* 🎨 Enhance UI/UX
* 🌐 Deploy with custom domain

---

## 🙌 Acknowledgement

This project is built as part of a **Deep Learning practice project** to understand sequence modeling using LSTM networks.

---

