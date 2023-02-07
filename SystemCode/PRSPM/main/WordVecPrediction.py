#!/usr/bin/env python
import os
import sys
import sklearn
from .TextPreprocessing import text_preprocessing
import pickle
import numpy as np
from tensorflow.keras import backend as K
from tensorflow.keras.losses import Loss
import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers import Conv1D
from tensorflow.keras.layers import AveragePooling1D
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Activation
from keras.preprocessing.sequence import pad_sequences

class custom_loss(Loss):
  weights = {}
  def __init__(self,weights):
    super().__init__()
    self.weights = weights

  def call(self, y_true, y_logit):
      '''
      Multi-label cross-entropy
      * Required "Wp", "Wn" as positive & negative class-weights
      y_true: true value
      y_logit: predicted value
      '''
      bce = tf.keras.losses.BinaryCrossentropy()
      loss = K.mean((self.weights[:,0]**(1-tf.cast(y_true, tf.float32)))*(self.weights[:,1]**(tf.cast(y_true, tf.float32)))*bce(y_true, y_logit), axis=-1)
      return loss

def createModel(modelname,num_classes,weight_matrix, vocab_size,max_length,class_weights_train):
    input = Input(shape=(max_length,))
    x = Embedding(vocab_size, 200,weights=[weight_matrix],  input_length=max_length,trainable=False)(input)
    
    x = Conv1D(filters=200, kernel_size=8,strides=2,padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(0.1)(x)

    x = Conv1D(filters=200, kernel_size=8,strides=2,padding='same')(x)
    x = Activation('relu')(x)
    x = AveragePooling1D(padding='same',pool_size=2)(x)

    x = Conv1D(filters=300, kernel_size=8,strides=2,padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(0.1)(x)

    x = Conv1D(filters=300, kernel_size=8,strides=2,padding='same')(x)
    x = Activation('relu')(x)
    x = AveragePooling1D(padding='same',pool_size=2)(x)

    x = Conv1D(filters=400, kernel_size=4,padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(0.1)(x)

    x = Conv1D(filters=400, kernel_size=4,padding='same')(x)
    x = Activation('relu')(x)
    x = AveragePooling1D(padding='same',pool_size=2)(x)

    x = Flatten()(x)
    x = Dense(100,activation='relu')(x)
    x = Dropout(0.25)(x)
    x = Dense(num_classes,activation='sigmoid')(x)
    model = Model(inputs=input,outputs=x)
    model.compile(loss=custom_loss(class_weights_train), optimizer='adam', metrics=[tf.keras.metrics.BinaryAccuracy(threshold=0.8),tf.keras.metrics.Precision(name='precision'),tf.keras.metrics.Recall(name='recall')])
    return model

def load_model():
    with open('./main/model/WordVec/class_weights_train.sav',"rb") as fp: 
        class_weights_train = pickle.load(fp)
    with open('./main/model/WordVec/weight_matrix.sav',"rb") as fp: 
        weight_matrix = pickle.load(fp)
    with open('./main/model/WordVec/WordVecTokenizer.pkl',"rb") as fp: 
        WordVecTokenizer = pickle.load(fp)
    vocab_size = len(WordVecTokenizer.word_index) + 1
    max_length = 2845
    WordVecDLModel = createModel('WordVecDLModel',41,weight_matrix,vocab_size,max_length,class_weights_train)
    WordVecDLModel.load_weights('./main/model/WordVec/WordVecDLModelWeight.hdf5')
    WordVecDLModel.compile(loss=custom_loss(class_weights_train), optimizer='adam', metrics=[tf.keras.metrics.BinaryAccuracy(threshold=0.8),tf.keras.metrics.Precision(name='precision'),tf.keras.metrics.Recall(name='recall')])
    return WordVecDLModel,WordVecTokenizer

def make_prediction(text):
    WordVecDLModel,WordVecTokenizer = load_model()
    df_symptom_text_list = [text]
    full_symptoms_tokenized = WordVecTokenizer.texts_to_sequences(df_symptom_text_list)
    input = pad_sequences(full_symptoms_tokenized, maxlen=2845, padding='post')
    WordVecPredicts = WordVecDLModel.predict(input)
    with open("./main/model/Bert/classes_41.txt", "rb") as fp: 
        classes_41 = pickle.load(fp)
    symptomlist = []
    thresh = 0.3
    for sentence, pred in zip(input, WordVecPredicts):
        pred = pred.tolist()
        for score in pred:
            if score>thresh:
                i = pred.index(score)
                symptom = classes_41[i]
                symptomlist.append(symptom)
    return symptomlist


    

            
    
    

