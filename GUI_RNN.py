import tkinter as tk
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import Sequential
from tensorflow.keras import regularizers
from PIL import Image, ImageTk
import webbrowser

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
	next_word = 1
	for _ in range(next_word):
		token_list = tokenizer.texts_to_sequences([test_seq])[0]
		token_list = pad_sequences([token_list], maxlen= max_sequences_len-1, padding='pre')
		predicted = model.predict_classes(token_list, verbose=0)
		output_word = ''
		for word, index in tokenizer.word_index.items():
			if index == predicted:
				output_word = word
				break
		test_seq += " " + output_word + " "
	return test_seq, output_word


def show_key(event):
	if event.keysym == 'space':
		test_seq, output_word = predict_model(search_text.get("1.0","end").strip())
		predict_word['text'] = output_word
	if event.keysym == 'Return':
		if predict_word['text'] !='':
			new_text = search_text.get("1.0","end").strip() + ' ' + predict_word['text']
			search_text.delete(1.0,"end")
			search_text.insert(1.0, new_text)
			predict_word['text'] = ''
			root.update()
		else:
			key_search = '+'
			words = search_text.get("1.0","end").strip().split()
			key_search = key_search.join(words)
			url = 'https://www.google.com/search?client=opera&sourceid=opera&ie=UTF-8&oe=UTF-8&q=' + key_search
			webbrowser.open(url)
			search_text.delete(1.0,"end")
			root.update()


root = tk.Tk()
root.title('Search Application')
root.geometry("+30+30")

canvas = tk.Canvas(root, height=700, width=1100)
canvas.pack()

frame_main = tk.Frame(root, bg='white')
frame_main.place(relx=0, rely=0, relwidth=1, relheight=1)

label_name = tk.Label(frame_main, font = ("Courier", 24, 'bold'), text = "Demo RNN - Nhom 9", fg = 'red', bg='white')
label_name.place(relx=0.1, rely=0.03)

load_image = Image.open("google.png")
render_image = ImageTk.PhotoImage(load_image)
google_logo = tk.Label(image = render_image, bg='white')
google_logo.place(relx = 0.1, rely=0.1, relwidth=0.8, relheight=0.3)


label_type_here = tk.Label(frame_main, font = ("Courier", 20), text = "Type Here", bg='white')
label_type_here.place(relx=0.1, rely=0.43)

next_word = tk.Label(frame_main, font = ("Courier", 20), text = 'Word predicted:', bg='white')
next_word.place(relx=0.53, rely=0.43)

predict_word = tk.Label(frame_main, font = ("Courier", 20, 'bold'), fg='red', bg='white')
predict_word.place(relx=0.78, rely=0.43)

search_text = tk.Text(frame_main, font =("Courier", 20), bd=3)
search_text.place(relx=0.1, rely=0.5, relwidth=0.8, relheight=0.25)

note = tk.Label(frame_main, font = ("Courier", 20), fg='blue',justify='left', bg='white', text='Note: \n- Press "Space" to get the next word.\n- Press "Enter" to fill in the predict word.\n- Press "Enter" again to search this on your web browser.')
note.place(relx=0.1, rely=0.75)

root.bind_all('<Key>', show_key)

root.mainloop()