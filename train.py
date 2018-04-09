from dna2vec.multi_k_model import MultiKModel
import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers import Embedding
from keras.layers import LSTM
import csv

filepath = 'pretrained/dna2vec-20161219-0153-k3to8-100d-10c-29320Mbp-sliding-Xat.w2v'
mk_model = MultiKModel(filepath)

df = pd.read_csv('train.csv')
vec8 = []
vec6 = []
print("Converting sequence to vectors")
for d in df['sequence']:	
	str8 = (d[:8])
	str6 = (d[8:])
	vec8.append(mk_model.vector(str8))
	vec6.append(mk_model.vector(str6))


X = np.zeros((2000, 2, 100))
Y = df['label']

for i in range(0,2000):
	for j in range(0, 2):
		if(j==0):
			X[i][j] = vec8[i]
		else:
			X[i][j] = vec6[i]
print(X.shape)			
print("Training model")
model = Sequential()
model.add(LSTM(100, input_shape = (2,100), return_sequences=True))
model.add(LSTM(64, return_sequences=True))
model.add(LSTM(32))
model.add(Dense(1))
model.add(Activation('sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam')
model.summary()


model.fit(X, Y, epochs = 10)

df = pd.read_csv('test.csv')

vec8 = []
vec6 = []
print("Converting sequence to vectors")
for d in df['sequence']:	
	str8 = (d[:8])
	str6 = (d[8:])
	vec8.append(mk_model.vector(str8))
	vec6.append(mk_model.vector(str6))

X = np.zeros((400, 2, 100))

for i in range(0,400):
	for j in range(0, 2):
		if(j==0):
			X[i][j] = vec8[i]
		else:
			X[i][j] = vec6[i]

Y = model.predict(X)
a = ["id", "prediction"]

writer = csv.writer(open("results.csv", "w"))
writer.writerow(a)
for i in range(0,400):
	a = [i,(int(Y[i].round()))]
	writer.writerow(a)