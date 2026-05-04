import streamlit as st
import pickle
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# ------------------------------
# Load resources (cached)
# ------------------------------
@st.cache_resource
def load_resources():
    model = load_model("lstm_model (1).h5")

    with open("tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)

    with open("max_len.pkl", "rb") as f:
        max_len = pickle.load(f)

    # Fast lookup dictionary
    index_to_word = {v: k for k, v in tokenizer.word_index.items()}

    return model, tokenizer, max_len, index_to_word


model, tokenizer, max_len, index_to_word = load_resources()

# ------------------------------
# Sampling (better predictions)
# ------------------------------
def sample_with_temperature(preds, temperature=0.8):
    preds = np.asarray(preds).astype("float64")
    preds = np.log(preds + 1e-8) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    return np.random.choice(len(preds), p=preds)


# ------------------------------
# Predict next word
# ------------------------------
def predict_next_word(text):
    text = text.lower().strip()

    seq = tokenizer.texts_to_sequences([text])[0]

    if len(seq) == 0:
        return ""

    seq = pad_sequences([seq], maxlen=max_len - 1, padding="pre")

    preds = model.predict(seq, verbose=0)
    pred_index = sample_with_temperature(preds[0], 0.8)

    return index_to_word.get(pred_index, "")


# ------------------------------
# Sentence auto-completion
# ------------------------------
def generate_text_auto(seed_text, max_words=30):
    seed_text = seed_text.lower().strip()

    for _ in range(max_words):
        next_word = predict_next_word(seed_text)

        if next_word == "":
            break

        seed_text += " " + next_word

        # Stop if sentence ends
        if next_word in [".", "!", "?"]:
            break

        # Safety stop (avoid infinite loop)
        if len(seed_text.split()) > 25:
            break

    return seed_text


# ------------------------------
# UI
# ------------------------------
st.set_page_config(page_title="AI Text Generator", layout="centered")

st.markdown("<h2>🧠 DL Word & Sentence Completion (LSTM)</h2>", unsafe_allow_html=True)
st.markdown("Generate the next word or complete a sentence automatically.")

mode = st.radio("Choose Mode:", ["Next Word", "Complete Sentence"])

user_input = st.text_input("✍️ Enter text:", placeholder="Type a sentence...")

if st.button("Generate"):
    if user_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        with st.spinner("Processing..."):

            if mode == "Next Word":
                word = predict_next_word(user_input)

                if word == "":
                    st.warning("Could not predict. Try different input.")
                else:
                    st.success(f"👉 Next Word: **{word}**")

            else:
                sentence = generate_text_auto(user_input)

                st.success(f"👉 Generated Sentence:\n\n{sentence}")


# ------------------------------
# Footer
# ------------------------------
st.markdown("---")
# st.caption("LSTM-based Sentence Completion using Streamlit")