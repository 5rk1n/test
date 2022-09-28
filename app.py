
from urllib.request import DataHandler
from flask import Flask,render_template,url_for, request
import pickle
import pandas as pd
import re
import nltk
import numpy as np
from nltk.tokenize import word_tokenize 
from nltk.stem.wordnet import WordNetLemmatizer
lemmatizing = WordNetLemmatizer()

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM , Dense, Dropout, Embedding, SpatialDropout1D
from keras.layers import Bidirectional
 


# load the model from disk
model_new = tf.keras.models.load_model("model.h5")

app = Flask(__name__)
@app.route('/')
def home():
	return render_template('home.html')

@app.route('/predict',methods=['POST'])
def predict():

	if request.method == 'POST':
		message = request.form['message']
		data = [message]
		my_prediction = model_new.predict(data)
        
	return render_template('result.html',prediction = my_prediction)



if __name__ == '__main__':
	app.run(debug=True)