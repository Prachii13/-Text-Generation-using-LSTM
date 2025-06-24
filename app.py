import streamlit as st
import numpy as np
import tensorflow as tf

model = tf.keras.models.load_model("model.h5")
char2idx = np.load("char2idx.npy", allow_pickle=True).item()
idx2char = np.load("idx2char.npy", allow_pickle=True).item()

SEQ_LENGTH = 100

st.title("✍️ LSTM Text Generator")
seed = st.text_area("Enter seed text (min 100 chars)")

if st.button("Generate"):
    seed = seed.lower().ljust(SEQ_LENGTH)[:SEQ_LENGTH]
    generated = seed
    for _ in range(300):
        x = np.zeros((1, SEQ_LENGTH, len(char2idx)))
        for t, char in enumerate(seed):
            if char in char2idx:
                x[0, t, char2idx[char]] = 1
        preds = model.predict(x)[0]
        next_idx = np.random.choice(len(preds), p=preds)
        next_char = idx2char[next_idx]
        generated += next_char
        seed = seed[1:] + next_char
    st.code(generated)
