
from keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

import tensorflow as tf

# load model 
model = load_model('model.h5')


def top_k_sampling(logits, k=10):
    values, indices = tf.math.top_k(logits, k)
    values /= tf.reduce_sum(values)
    chosen_index = tf.random.categorical(tf.math.log(values), 1)[0, 0]
    return indices[0, chosen_index].numpy()

def generate_text_top_k(seed_text, num_words, k=10):
    generated_text = seed_text
    tokenizer = Tokenizer()
    max_sequence_len=2030
    for _ in range(num_words):
        token_list = tokenizer.texts_to_sequences([seed_text])[0]
        token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding="pre")
        predicted = model.predict(token_list, verbose=0)

        predicted_word_index = top_k_sampling(predicted, k)
        predicted_word = tokenizer.index_word[predicted_word_index]

        seed_text += " " + predicted_word
        generated_text += " " + predicted_word
    return generated_text


generated_text_top_k = generate_text_top_k("First of all, the company took my money", num_words=10, k=10)
print(generated_text_top_k)



