#!/usr/bin/env python
import os
import sys
import sklearn
from .TextPreprocessing import text_preprocessing
import pickle
import numpy as np



def load_model():
    with open('./main/model/TFIDF/TfidfVectorizer.pkl',"rb") as fp: 
        tfidfVectorizer = pickle.load(fp)
    with open('./main/model/TFIDF/MultinomialModel.sav',"rb") as fp: 
        multiNomialModel = pickle.load(fp)
    with open('./main/model/TFIDF/SVMModel.sav',"rb") as fp: 
        SVMModel = pickle.load(fp)
    return tfidfVectorizer,multiNomialModel,SVMModel

def make_prediction(text):
    tfidfVectorizer,multiNomialModel,SVMModel = load_model()
    preprocessedText    = np.asarray([text_preprocessing(text)])
    X = tfidfVectorizer.transform(preprocessedText)
    svm_predicts = SVMModel(X)
    with open("./main/model/Bert/classes_41.txt", "rb") as fp: 
        classes_41 = pickle.load(fp)
    symptomlist = []
    for i,value in enumerate(svm_predicts.tolist()):
        if value>0.5:
            symptom = classes_41[i]
            symptomlist.append(symptom)
    return symptomlist
    

            
    
    

