#https://data-flair.training/blogs/python-chatbot-project/

import nltk 
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
import pickle
import numpy as np

from tensorflow.keras.models import load_model
import json 
import random ,os
pathtochat = os.path.join(os.getcwd(),'chat_bot_final') 

model = load_model(os.path.join(pathtochat,'chat_model.h5'))
intents = json.loads(open(os.path.join(pathtochat,'intents.json'),encoding="utf8").read())
words = pickle.load(open(os.path.join(pathtochat,'words.pkl'),'rb'))
classes = pickle.load(open(os.path.join(pathtochat,'classes.pkl'),'rb'))


def clean_up_sentence(sent):
    sentence_words = nltk.word_tokenize(sent)
    sentence_words = [lemmatizer.lemmatize(w.lower()) for w in sentence_words]
    return sentence_words

def bow(sentence , words , show_details = True):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for s in sentence_words:
        for i , w in enumerate(words):
            if w == s:
                bag[i] = 1
                if show_details:
                    print('found in beg: %s' % w)
    return (np.array(bag))


def predict_class(sentence , model):
    p = bow(sentence , words, show_details = False)
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i,r] for i,r in enumerate(res) if r > ERROR_THRESHOLD ]
    results.sort(key=lambda x: x[1] , reverse = True)
    return_list = []
    for r in results:
        return_list.append({'intent':classes[r[0]],'probability':str(r[1])})
    return return_list


def getResponse(ints,intents_json):
    result =''
    tag = ints[0]['intent']
    list_of_intents = intents_json['intents']
    
    for i in list_of_intents:
        if(i['tag']==tag):
            result = random.choice(i['responses'])
            break
    return result

def chatbot_response(msg):
    ints = predict_class(msg,model)
    res = getResponse(ints,intents)
    
    return res


