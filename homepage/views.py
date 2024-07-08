from django.shortcuts import render
import pickle as pkl
import numpy as np
import random
import nltk
import json
from keras.models import load_model
from nltk.stem import WordNetLemmatizer
lemmatise = WordNetLemmatizer()

data_file=open("C:\\Users\\Sreerag\\Data Science\\Internship\\Datasets\\intents.json").read()
intents=json.loads(data_file)
model = load_model('chatbot_model.h5')
words = pkl.load(open('words.pkl','rb'))
classes=pkl.load(open('classes.pkl','rb'))

# Create your views here.

def homepage(req):
    return render(req,'homepage.html')

def predict(req):
    input = req.POST.get('user_input')

    def clean_up_sentence(sentense):
        sentense_words=nltk.word_tokenize(sentense)
        sentense_words=[lemmatise.lemmatize(word.lower()) for word in sentense_words]
        return sentense_words

    def bow(sentense,words,show_details=False):
        sentense_words=clean_up_sentence(sentense)
        bag=[0]*len(words)
        for s in sentense_words:
            for i,w in enumerate(words):
                if w==s:
                    bag[i]=1
                    if show_details:
                        print('found in bag')
        return np.array(bag)

    def predict_class(sentense,model):
        p=bow(sentense,words,show_details=False)
        res=model.predict(np.array([p]))[0]
        ERROR_THRESHOLD=0.25
        results=[[i,r] for i,r in enumerate(res) if r>ERROR_THRESHOLD]
        results.sort(key=lambda x:x[1],reverse=True)
        return_list=[]
        for r in results:
            return_list.append({'intent':classes[r[0]],'probability':str(r[1])})
        return return_list
    
    def response(ints,intents_json):
        tags=ints[0]['intent']
        list_of_intent=intents_json['intents']
        for i in list_of_intent:
            if (i['tag']==tags):
                result=random.choice(i['responses'])
        return result

    ints=predict_class(input,model)
    response_msg=response(ints,intents)
    return render(req,'homepage.html',{'response':response_msg})