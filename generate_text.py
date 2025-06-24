import tensorflow as tf
import numpy as np

model = tf.keras.models.load_model("model.h5")
char2idx = np.load("char2idx.npy", allow_pickle=True).item()
idx2char = np.load("idx2char.npy", allow_pickle=True).item()

SEQ_LENGTH = 100

seed = input("Enter seed text (100 chars): ").lower()[:SEQ_LENGTH].ljust(SEQ_LENGTH)
generated = seed

for _ in range(400):
    x = np.zeros((1, SEQ_LENGTH, len(char2idx)))
    for t, char in enumerate(seed):
        if char in char2idx:
            x[0, t, char2idx[char]] = 1
    preds = model.predict(x)[0]
    next_idx = np.random.choice(len(preds), p=preds)
    next_char = idx2char[next_idx]
    generated += next_char
    seed = seed[1:] + next_char

print("\n📝 Generated Text:\n")
print(generated)
