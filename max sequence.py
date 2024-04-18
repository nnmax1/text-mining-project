import re
import csv
import string
import pickle
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt

import tensorflow as tf
import csv

corpus = []

with open('amazon_dataset.csv') as f:
    reader = csv.reader(f, delimiter=',')
    next(reader)        
    for row in reader:
        #print(row[2])
        corpus.append(row[2])
        
#print(len(corpus))
#print(corpus[:3])

# clean text
import string
def text_cleaner(text):
    text = re.sub(r'\s+\n+', ' ', text)
    text = re.sub(r'[^a-zA-Z0-9\.]', ' ', text)
    text = "".join(car for car in text if car not in string.punctuation).lower()
    text = text.encode("utf8").decode("ascii",'ignore')
    return text

corpus = [text_cleaner(line) for line in corpus]

# tokenize data
from tensorflow.keras.preprocessing.text import Tokenizer

tokenizer = Tokenizer()
tokenizer.fit_on_texts(corpus)
word_index = tokenizer.word_index
total_words = len(word_index) + 1
total_words



input_sequences =[]
for sentence in corpus[12000:20000]:
    token_list = tokenizer.texts_to_sequences([sentence])[0]
    for i in range(1, len(token_list)):
        n_gram_sequence = token_list[:i+1]
        input_sequences.append(n_gram_sequence)



from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow
# length of longest sequence
max_sequence_len = max([len(x) for x in input_sequences])
input_sequences = np.array(pad_sequences(input_sequences, 
                                         maxlen=max_sequence_len, 
                                         padding='pre'))

print(max_sequence_len)

