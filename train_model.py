import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Embedding
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
import csv

# Example data preprocessing (replace with your own data preparation code)
# get Amazon review data
reviews=[]
with open('amazon_dataset.csv') as f:
    reader = csv.reader(f, delimiter=',')
    next(reader)        
    for row in reader:
        #print(row[2])
        reviews.append(row[2])

# Tokenization
from tensorflow.keras.preprocessing.text import Tokenizer

tokenizer = Tokenizer()
tokenizer.fit_on_texts(reviews)
sequences = tokenizer.texts_to_sequences(reviews)
word_index = tokenizer.word_index
vocab_size = len(word_index) + 1
# Padding sequences
max_seq_length = max([len(seq) for seq in sequences])
sequences = pad_sequences(sequences, maxlen=max_seq_length, padding='post')

print("max seq",max_seq_length)
# Prepare input and output data
X = sequences[:, :-1]
y = to_categorical(sequences[:, -1], num_classes=vocab_size)


# https://github.com/stanfordnlp/GloVe?tab=readme-ov-file
glove_path = 'glove.42B.300d.txt'
embeddings_index = {}
with open(glove_path,
          encoding="utf8") as glove:
    for line in glove:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    glove.close()

embedding_matrix = np.zeros((vocab_size, 300))
for word, index in tokenizer.word_index.items():
    if index > vocab_size - 1:
        break
    else:
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[index] = embedding_vector


# Define LSTM model
model = Sequential()
model.add(Embedding(vocab_size, 300, weights = [embedding_matrix],input_length=max_seq_length-1))
model.add(LSTM(100))
model.add(Dense(vocab_size, activation='softmax'))


# Compile model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# Train model
model.fit(X, y, epochs=1, verbose=1)
model.save('model.h5')
    
 

