import nltk
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
import json
import pickle

import numpy as np
from tensorflow import keras 
from keras.models import Sequential
from keras.layers import Dense ,Activation , Dropout
from tensorflow.keras.optimizers import SGD
import random


words = []
classes = []
documents = []
ignore_words = ['!' , '?']
data_file = open('intents.json',encoding="utf8").read()
intents = json.loads(data_file)

# add each question's word to list  
for intent in intents['intents']:
    for pattern in intent['patterns']:
        w = nltk.word_tokenize(pattern)
        words.extend(w)
        documents.append((w,intent['tag']))
    if intent['tag'] not in classes:
        classes.append(intent['tag'])
print(classes)
print('documents')
print('---------')
print(documents[:5])
print('words')
print('-----')
print(words[:5])

words = [lemmatizer.lemmatize(w.lower()) for w in words if  w not in ignore_words]
# get red of dublications 
words = sorted(list(set(words)))
classes = sorted(list(set(classes)))

print(len(documents),' documents')
print(len(classes),' classes')
print(len(words),' unique lemmatized words')

# save the list of all ward and classes in files
pickle.dump(words, open('words.pkl','wb'))
pickle.dump(classes, open('classes.pkl','wb'))



training = []
output_empty = [0] * len(classes)
## 
# make bag of word for each doc
for doc in documents:
    bag = []
    pattern_words = doc[0]
    pattern_words = [lemmatizer.lemmatize(w.lower()) for w in pattern_words if w not in ignore_words]
                   
    for w in words:
        bag.append(1) if w in pattern_words else bag.append(0)
        
    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1
    
    training.append([bag , output_row])



# random the training sampile
random.shuffle(training)
training = np.array(training)

train_x = list(training[:,0])
train_y = list(training[:,1])


model = Sequential()
model.add(Dense(128,input_shape = (len(train_x[0]),), activation = 'relu' ))
model.add(Dropout(0.5))
model.add(Dense(64, activation = 'relu' ))
model.add(Dropout(0.5))
model.add(Dense( len(train_y[0]), activation = 'softmax' ))

sgd = SGD(learning_rate=0.01, decay=1e-6 , momentum=0.9,nesterov = True)
model.compile(loss='categorical_crossentropy',optimizer = sgd,metrics=['accuracy'])

# train the model and save 
hist = model.fit(np.array(train_x),np.array(train_y),epochs=200 , batch_size = 5 , verbose = 1)
model.save('chat_model.h5',hist)