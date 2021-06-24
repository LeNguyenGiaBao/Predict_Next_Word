from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
from tensorflow.keras import regularizers
from tensorflow.keras.callbacks import ModelCheckpoint
import tensorflow.keras.utils as ku
import numpy as np
import matplotlib.pyplot as plt


data = open('data_vtc_giao_duc.txt', encoding="utf8").read()
corpus = data.lower().split('\n')

tokenizer = Tokenizer()
tokenizer.fit_on_texts(corpus)
total_words = len(tokenizer.word_index) + 1
max_sequences_len=33

model = Sequential()
model.add(Embedding(total_words, 100, input_length= max_sequences_len - 1))
model.add(Bidirectional(LSTM(150, return_sequences=True)))
model.add(Dropout(0.2))
model.add(LSTM(100))
model.add(Dense(total_words/2, activation= 'relu', kernel_regularizer=regularizers.l2(0.01)))
model.add(Dense(total_words, activation='softmax'))
model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])

model.load_weights('vtc_giao_duc_data.h5')

def predict_model(test_seq):
	next_word = 6

	for _ in range(next_word):
		token_list = tokenizer.texts_to_sequences([test_seq])[0]
		# print(token_list)
		token_list = pad_sequences([token_list], maxlen= max_sequences_len-1, padding='pre')
		# print(token_list)
		predicted = model.predict_classes(token_list, verbose=0)
		# print(predicted)
		output_word = ''
		for word, index in tokenizer.word_index.items():
			if index == predicted:
				output_word = word
				break
		test_seq += " " + output_word
	return test_seq

print(predict_model('điểm chuẩn'))
