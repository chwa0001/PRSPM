#!/usr/bin/env python

import pickle 
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import concatenate
from tensorflow.keras import regularizers
from tensorflow.keras import optimizers
from tensorflow.keras.layers import Activation
import numpy as np
import pandas as pd

def createModel():
    lr_schedule = optimizers.schedules.ExponentialDecay(initial_learning_rate=0.005,decay_steps=15,decay_rate=0.3)
    optmz       = optimizers.Adam(learning_rate=lr_schedule)                                     
    
    Lin   = Input(shape=(59), name = 'leftIn')
    Lx       = Dense(250,activation='relu')(Lin)
    Lx       = BatchNormalization(momentum=0.99, epsilon=0.001, center=True, scale=True)(Lx)
    Lx       = Dense(100,kernel_regularizer=regularizers.l2(0.01))(Lx)
    Lx       = BatchNormalization(momentum=0.99, epsilon=0.001, center=True, scale=True)(Lx)
    Lx       = Activation('relu')(Lx)
    Lx       = Dense(1,activation='relu',kernel_regularizer=regularizers.l2(0.15))(Lx)
    
    Rin      = Input(shape=(4), name = 'RightIn')
    Rx       = Dense(30,activation='relu',kernel_regularizer=regularizers.l2(0.1))(Rin)
    Rx       = BatchNormalization(momentum=0.99, epsilon=0.01, center=True, scale=True)(Rx)
    Rx       = Dense(1,activation='relu',kernel_regularizer=regularizers.l2(0.1))(Rx)

    x       = concatenate([Lx,Rx], axis=-1)
    x       = Dense(30,activation='relu',kernel_regularizer=regularizers.l2(0.25))(x)
    x       = BatchNormalization(momentum=0.99, epsilon=0.01, center=True, scale=True)(x)
    x       = Dense(1,activation='sigmoid',kernel_regularizer=regularizers.l2(0.5))(x)
    
    model = Model(inputs=[Lin,Rin],outputs=x)
    model.load_weights('./main/model/Hospitalised/Covid-Hospitalization-Prediction-ML-2input_proba_250_cw.5_15.hdf5')
    model.compile(loss=tf.keras.losses.BinaryCrossentropy(),optimizer=optmz,metrics=[tf.keras.metrics.BinaryAccuracy(),tf.keras.metrics.BinaryCrossentropy(),tf.keras.metrics.FalsePositives(),tf.keras.metrics.FalseNegatives()])
     
    return model
    
class CovidHospitalizationPrediction():
    def __init__(self,symptomsClass):
        with open('./main/model/Hospitalised/lgb_model_1024.sav',"rb") as fp: 
            self.lgb = pickle.load(fp)
        with open('./main/model/Hospitalised/xgb_model_1024.sav',"rb") as fp: 
            self.xgb = pickle.load(fp)
        with open('./main/model/Hospitalised/rfc_model_1024.sav',"rb") as fp: 
            self.rfc = pickle.load(fp)
        with open('./main/model/Hospitalised/lr_model_1024.sav',"rb") as fp: 
            self.lr = pickle.load(fp)
        with open('./main/model/Hospitalised/sc.sav',"rb") as fp: 
            self.sc = pickle.load(fp)
        with open('./main/model/Hospitalised/ohe.sav',"rb") as fp: 
            self.ohe = pickle.load(fp)
        with open('./main/model/Hospitalised/imputer.sav',"rb") as fp: 
            self.imputer = pickle.load(fp)
        with open('./main/model/Hospitalised/imputer_cat.sav',"rb") as fp: 
            self.imputer_cat = pickle.load(fp)

        self.symp_col = list(symptomsClass)
        self.modelCovidHospitalizationPrediction = createModel()

    def makePrediction(self,dictData):
        num_col = ['AGE_YRS', 'NUMDAYS']
        bool_col = ['L_THREAT', 'DISABLE', 'BIRTH_DEFECT','OFC_VISIT']
        cat_col = ['SEX', 'VAX_MANU', 'VAX_DOSE_SERIES']
        symptoms = ['symptoms']
        
        vaers = pd.DataFrame(data=dictData,columns=(num_col+bool_col+cat_col+symptoms))

        for symptom_column in self.symp_col: 
            vaers[symptom_column] = 1 if symptom_column in symptoms else 0

        vaers.drop(['symptoms'],axis=1,inplace=True)

        X_bool = vaers[self.symp_col+bool_col].copy()
        X_bool_col = self.symp_col+bool_col
        X_bool = X_bool.to_numpy()
        
        X_cat = vaers[cat_col].copy()
        X_cat = self.imputer_cat.transform(X_cat)
        X_cat = self.ohe.transform(X_cat)
        X_cat = X_cat.toarray()
        ohe_categories = self.ohe.categories_

        X_cat_col = []
        for i,c in enumerate(cat_col):
            for j in ohe_categories[i]:
                X_cat_col.append(f'{c}__{j}')

        X_num = self.imputer.transform(vaers[num_col])
        X_num = self.sc.transform(X_num)

        X_num_col = num_col

        data_frame = pd.DataFrame(np.concatenate((X_num,X_cat,X_bool),axis=1),columns=X_num_col+X_cat_col+X_bool_col)
        feature_column_names = data_frame.columns
        print(len(feature_column_names))
        del X_num, X_cat, X_bool

        X = data_frame[X_num_col+X_cat_col+X_bool_col].values
        
        proba = [self.rfc.predict_proba(X)[:, 1][0],self.lgb.predict_proba(X)[:, 1][0] , self.xgb.predict_proba(X)[:, 1][0], self.lr.predict_proba(X)[:, 1][0]]
        predicts_train    = self.modelCovidHospitalizationPrediction.predict([np.array(X).reshape(1,-1),np.array(proba).reshape(1,-1)])    
        return predicts_train

            
    
    

