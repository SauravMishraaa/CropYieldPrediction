# -*- coding: utf-8 -*-
"""
Created on Sun Jul  9 12:51:24 2023

@author: pc
"""

from fastapi import FastAPI
import uvicorn
from predictions import plantyieldprediction
from fastapi.encoders import jsonable_encoder
import numpy as np
import pickle
import sklearn
import joblib
app=FastAPI()

pickle_in = open("NBClassifier.pkl","rb")
classifier=pickle.load(pickle_in)

#job=joblib.load('NB_Model.obj',mmap_mode=None)

@app.get('/')   
def index():
    return {'message': 'Hello'}


@app.post('/predict')
def plantyield(data:plantyieldprediction):
    #print(data)
    data=data.dict()

    N=data['N']
    K=data['K']
    P=data['P']
    temperature=data['temperature']
    humidity=data['humidity']
    ph=data['ph']
    rainfall=data['rainfall']
    
    #import skops.io as sio
    #obj = sio.dumps(clf)
    
    #unknown_types = sio.get_untrusted_types(data=obj)
    #clf = sio.loads(obj, trusted=unknown_types)
    #clf = sio.loads(obj, trusted=True)
    
    prediction=classifier.predict([[N,K,P,temperature,humidity,ph,rainfall]])
    jsonable_prediction=jsonable_encoder(prediction)
    return jsonable_prediction
    return prediction

if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)