# -*- coding: utf-8 -*-
"""
Created on Sun Oct 24 08:01:15 2021

@author: antonia
"""

#%%

import matplotlib
import sklearn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import datetime as dt
import sklearn.metrics as metrics
import pickle

from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.utils import shuffle

import sys,random
import sklearn.metrics as metrics
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import classification_report, roc_auc_score, precision_score, recall_score, f1_score, log_loss, confusion_matrix
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc


#%%
def implt(img):
    plt.figure()
    plt.imshow(img)
    plt.axis('off')

print(implt)

plt.style.use('seaborn')                   # if want to use the default style, set 'classic'
plt.rcParams['ytick.right']     = True
plt.rcParams['ytick.labelright']= True
plt.rcParams['ytick.left']      = False
plt.rcParams['ytick.labelleft'] = False
plt.rcParams['figure.figsize']  = [7,7]   # Set the figure size to be 7 inch for (width,height)

print("Matplotlib setup completes.")

#%%

DIR = r'C:\Users\antonia\OneDrive - National University of Singapore\Documents\Covid Project\\'

vaers = pd. read_csv(DIR+"VAERS_VAX_SYMP_T100.csv",index_col=0)
col_types = pd.read_csv(DIR+"columntype_VAERS_VAX_SYMP_T100.csv",index_col=0)
vaers['ER'] = vaers['ER'].apply(lambda a : 1 if 'Y' in a else 0)
vaers['RECOVD'] = vaers['RECOVD'].apply(lambda a : 1 if 'Y' in str(a) else 0)
vaers.NUMDAYS[vaers.NUMDAYS>20] = 20


#%%

num_col = ['AGE_YRS', 'NUMDAYS']
bool_col = ['L_THREAT', 'DISABLE', 'BIRTH_DEFECT','OFC_VISIT']
cat_col = ['SEX', 'VAX_MANU', 'VAX_DOSE_SERIES']

#%% Only using the symptoms extracted by the BERT

symp = pd.read_pickle(DIR+"ben_1014_data_complete2r.csv",)
symp_col = list(symp.columns)[2:]
symptoms_col = [col for col in symptoms_col if col in symp_col]
print('length of symp found == length of symp given :   ', len([col for col in symptoms_col if col in symp_col])==len(symp_col))

#%% Additional Clean up 


# Vaccination dose more than 3 consider as 3+

vaers.VAX_DOSE_SERIES = [i if i in ['1','2','3','UNK'] else '3+' for i in vaers.VAX_DOSE_SERIES]
vaers.VAX_DOSE_SERIES = [i if i not in ['N','UNK'] else 'UNK' for i in vaers.VAX_DOSE_SERIES]

#%%


# def tidyup_dataframe_bool_cat_impute_future_dataframe(hist_cur_ill_col, allergy_col, bool_col, cat_col, num_col):

print(f"-"*50,"Bool","-"*50); print()

X_bool = vaers[symptoms_col+bool_col].copy()
X_bool_col = symptoms_col+bool_col
X_bool=X_bool.to_numpy()

print(f"Length of col match the dimension of the array: {len(X_bool_col) == len(X_bool[1])}")

#-----------------------------------------------------------------------------------------------------------
print();print(f"-"*50,"Categorical","-"*50); print()


X_cat = vaers[cat_col].copy()
imputer_cat = SimpleImputer(missing_values=np.nan, strategy='constant', fill_value='N')
X_cat = imputer_cat.fit_transform(X_cat)
ohe = OneHotEncoder(handle_unknown='ignore') #,drop='first')
X_cat = ohe.fit_transform(X_cat)
X_cat = X_cat.toarray()
feature_col = ohe.get_feature_names()
ohe_categories = ohe.categories_

X_cat_col = []
for i,c in enumerate(cat_col):
#     print(i,c)
    for j in ohe_categories[i]:
        X_cat_col.append(f'{c}__{j}')

print(X_cat_col)
# print(X_cat[:2])
print(f"Length of col match the dimension of the array: {len(X_cat_col) == len(X_cat[1])}")

#-----------------------------------------------------------------------------------------------------------
print();print(f"-"*50,"Numerical","-"*50); print()

imputer = SimpleImputer(strategy='mean')
X_num = imputer.fit_transform(vaers[num_col])
sc = StandardScaler()
X_num = sc.fit_transform(X_num)

X_num_col = num_col
print(X_num_col)
# print(X_num[:2])
print(f"Length of col match the dimension of the array: {len(X_num_col) == len(X_num[1])}")

#-----------------------------------------------------------------------------------------------------------
print();print(f"-"*50,"Drop columns and combine","-"*50); print()

drop_least_imp = []
data_frame = pd.DataFrame(np.concatenate((X_num,X_cat,X_bool),axis=1),columns=X_num_col+X_cat_col+X_bool_col).drop(drop_least_imp,axis=1)


predicted_class_name = ['HOSPITAL']

# drop_col = ['HOSPDAYS','DIED','ER']#'HOSPITAL']
# data_frame.drop(drop_col,axis=1,inplace=True)

feature_column_names = data_frame.columns

print(X_num.shape, X_cat.shape, X_bool.shape,data_frame.shape)
print(predicted_class_name, feature_column_names)

#%%

#%%

del X_num, X_cat, X_bool

#%%

final_feature_col = [columns for columns in list(feature_column_names) if ('V_ADMINBY' not in columns) and ('HOSPITAL'!=columns) and ('RECOVD'!=columns)]

X = data_frame[final_feature_col].values
y = vaers[predicted_class_name].values.ravel()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=21)

print(final_feature_col)


#%%

def print_scores (y, y_pred, y_proba):
    precision = precision_score(y, y_pred)
    recall = recall_score(y, y_pred)
    f1 = f1_score(y, y_pred)
    test_acc = metrics.accuracy_score(y, y_pred)
    auc = roc_auc_score(y, y_proba)
    logloss = log_loss(y, y_pred)

    # print ("test_acc :{0:.2f}".format(test_acc))
    # print ("precision:{0:.2f}".format(precision))
    # print ("recall   :{0:.2f}".format(recall))
    # print ("f1       :{0:.2f}".format(f1))
    # print ("auc      :{0:.2f}".format(auc))
    # print ("logloss  :{0:.2f}".format(logloss))
    # print (test_acc, precision, recall, f1, auc, logloss)

def print_model_evaluation (ml_model, score_type, rf_accuracy, y, y_predict, y_proba, y_pred_all, y_proba_all, save_data, description):
  print (f"{ml_model} {score_type} accuracy: {rf_accuracy}")
  print ("{0}".format(metrics.classification_report(y, y_predict, labels=[1, 0],digits=4)))
  print(confusion_matrix(y, y_predict))
  print ()
  print(f'{score_type} Accuracies')
  print_scores (y, y_predict, y_proba)

  # Compute ROC curve and ROC area for each class
  fpr = dict()
  tpr = dict()
  roc_auc = dict()
  fpr, tpr, _ = roc_curve(y, y_proba)
  roc_auc = auc(fpr, tpr)

  plt.figure()
  lw = 2
  plt.plot(fpr, tpr, color='darkorange',
          lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
  plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
  plt.xlim([0.0, 1.0])
  plt.ylim([0.0, 1.0])
  plt.xlabel('False Positive Rate')
  plt.ylabel('True Positive Rate')
  plt.title(f'{score_type} Receiver operating characteristic')
  plt.legend(loc="lower right")
  plt.show()

  print ();print ();print ("-"*50);print ()
  
  model_name = ml_model+score_type

  y_pred_all[model_name] = y_predict
  y_proba_all[model_name] = y_proba
  tn, fp, fn, tp = confusion_matrix(y, y_predict).ravel()
  save_data.loc[save_data.shape[0]] = description + [tn, fp, fn, tp]
  save_data.to_csv(DIR + "Model_training_score.csv")


#%%

#collection of all the predictions
y_pred_all = {}
y_proba_all = {}
save_data = pd.DataFrame(columns=['date','description1','description2','test/train','model','model_des','train_time','TN','FP','FN','TP'])

# save_data = pd.read_csv(DIR + "Model_training_score.csv",index_col=0)

#%%

from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
import lightgbm as lgb
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.naive_bayes import MultinomialNB

#%%

def RandomForestModel_TrainTest(X_train, X_test, y_train, y_test, description1, description2):
  model_name = "rfc"
  # description1 = "SMOTENC random 21"
  # description2 = "Allergies+Bool+Num"
  model_description = "n_estimators=100, min_samples_split=2, max_depth=8, criterion='gini', random_state=42"

  time_start = time.perf_counter()

  #model training
  rf_model = RandomForestClassifier(n_estimators=100, min_samples_split=2, max_depth=8, criterion='gini', random_state=42)
  rf_model.fit(X_train, y_train.ravel())

  time_elapsed = (time.perf_counter() - time_start)
  print (f"Time and resources: {time_elapsed}")

  #training prediction
  y_train_pred = rf_model.predict(X_train)
  y_train_proba = rf_model.predict_proba(X_train)[:, 1]

  importances = rf_model.feature_importances_

  #test prediction
  y_test_pred = rf_model.predict(X_test)
  y_test_proba = rf_model.predict_proba(X_test)[:, 1]

  #save model
  filename = f'{model_name}_model_1024.sav'
  pickle.dump(rf_model, open(DIR+filename, 'wb'))
  
  #print accuracy
  print_model_evaluation(model_name,"Train",metrics.accuracy_score(y_train, y_train_pred),y_train, y_train_pred, y_train_proba, y_pred_all, y_proba_all, save_data,
                        [dt.datetime.now(), description1, description2, "train", model_name,model_description,time_elapsed])
  print_model_evaluation(model_name,"Test",metrics.accuracy_score(y_test, y_test_pred),y_test, y_test_pred, y_test_proba,y_pred_all,y_proba_all,save_data,
                        [dt.datetime.now(),description1, description2, "test", model_name,model_description,time_elapsed])

def lightgbm_TrainTest(X_train, X_test, y_train, y_test, description1, description2):
  model_name = "lgb"
  # description1 = "SMOTE random 21"
  # description2 = "Allergies+Bool+Num"
  model_description = "default"

  time_start = time.perf_counter()

  #model training
  lgbm = lgb.LGBMClassifier(random_state = 21)
  lgbm.fit(X_train, y_train.ravel())

  time_elapsed = (time.perf_counter() - time_start)
  print (f"Time and resources: {time_elapsed}")

  #training prediction
  y_train_pred = lgbm.predict(X_train)
  y_train_pred = np.where(y_train_pred > 0.5, 1, 0)
  y_train_proba = lgbm.predict_proba(X_train)[:, 1]

  importances = lgbm.feature_importances_

  # test prediction
  y_test_pred = lgbm.predict(X_test)
  y_test_pred = np.where(y_test_pred > 0.5, 1, 0)
  y_test_proba = lgbm.predict_proba(X_test)[:, 1]

  #save model
  filename = f'{model_name}_model_1024.sav'
  pickle.dump(lgbm, open(DIR+filename, 'wb'))
  
  #print accuracy
  print_model_evaluation(model_name,"Train",metrics.accuracy_score(y_train, y_train_pred),y_train, y_train_pred, y_train_proba, y_pred_all, y_proba_all, save_data,
                        [dt.datetime.now(), description1, description2, "train", model_name,model_description,time_elapsed])
  print_model_evaluation(model_name,"Test",metrics.accuracy_score(y_test, y_test_pred),y_test, y_test_pred, y_test_proba,y_pred_all,y_proba_all,save_data,
                        [dt.datetime.now(),description1, description2, "test", model_name,model_description,time_elapsed])

def xgboost_TrainTest(X_train, X_test, y_train, y_test, description1, description2):
  model_name = "xgb"
  # description1 = "SMOTE random 21"
  # description2 = "Allergies+Bool+Num"
  model_description = "default"

  time_start = time.perf_counter()

  # Create model and train model
  xgb_model = XGBClassifier(learning_rate =0.1, n_estimators=140, max_depth=11,
     min_child_weight=1, gamma=0, subsample=0.8, colsample_bytree=0.8,
     objective= 'binary:logistic', nthread=4, scale_pos_weight=1,seed=21,use_label_encoder=False, eval_metric='auc')
  xgb_model.fit(X_train, y_train)

  time_elapsed = (time.perf_counter() - time_start)
  print (f"Time and resources: {time_elapsed}")

  # training prediction
  y_train_pred = xgb_model.predict(X_train)
  y_train_proba = xgb_model.predict_proba(X_train)[:, 1]

  importances = xgb_model.feature_importances_
  print(importances)  

  # test prediction
  y_test_pred = xgb_model.predict(X_test)
  y_test_proba = xgb_model.predict_proba(X_test)[:, 1]

  #save model
  filename = f'{model_name}_model_1024.sav'
  pickle.dump(xgb_model, open(DIR+filename, 'wb'))
  
  #print accuracy
  print_model_evaluation(model_name,"Train",metrics.accuracy_score(y_train, y_train_pred),y_train, y_train_pred, y_train_proba, y_pred_all, y_proba_all, save_data,
                        [dt.datetime.now(), description1, description2, "train", model_name,model_description,time_elapsed])
  print_model_evaluation(model_name,"Test",metrics.accuracy_score(y_test, y_test_pred),y_test, y_test_pred, y_test_proba,y_pred_all,y_proba_all,save_data,
                        [dt.datetime.now(),description1, description2, "test", model_name,model_description,time_elapsed])


def LogisticRegression_TrainTest(X_train, X_test, y_train, y_test, description1, description2):
  model_name = "lr"
  # description1 = "SMOTE random 21"
  # description2 = "Allergies+Bool+Num"
  model_description = "max_iter=1000"

  time_start = time.perf_counter()

  # Create and train model
  lr=LogisticRegression(max_iter=100000)
  lr.fit(X_train, y_train)

  time_elapsed = (time.perf_counter() - time_start)
  print (f"Time and resources: {time_elapsed}")

  # training prediction
  y_train_pred = lr.predict(X_train)
  y_train_proba = lr.predict_proba(X_train)[:, 1]

  # importances = lr.coef_[0]

  # test prediction
  y_test_pred = lr.predict(X_test)
  y_test_proba = lr.predict_proba(X_test)[:, 1]

  #save model
  filename = f'{model_name}_model_1024.sav'
  pickle.dump(lr, open(DIR+filename, 'wb'))
  
  #print accuracy
  print_model_evaluation(model_name,"Train",metrics.accuracy_score(y_train, y_train_pred),y_train, y_train_pred, y_train_proba, y_pred_all, y_proba_all, save_data,
                        [dt.datetime.now(), description1, description2, "train", model_name,model_description,time_elapsed])
  print_model_evaluation(model_name,"Test",metrics.accuracy_score(y_test, y_test_pred),y_test, y_test_pred, y_test_proba,y_pred_all,y_proba_all,save_data,
                        [dt.datetime.now(),description1, description2, "test", model_name,model_description,time_elapsed])

def MultinomialNB_TrainTest(X_train, X_test, y_train, y_test, description1, description2):
    model_name = "mnb"
    # description1 = "SMOTE random 21"
    # description2 = "Allergies+Bool+Num"
    model_description = "algorithm = 'brute', n_jobs=-1"
    
    time_start = time.perf_counter()
    
    # Create and train model
    mnb = MultinomialNB()
    mnb.fit(X_train, y_train)
    
    time_elapsed = (time.perf_counter() - time_start)
    print (f"Time and resources: {time_elapsed}")
    
    # training prediction
    y_train_pred = mnb.predict(X_train)
    y_train_proba = mnb.predict_proba(X_train)[:, 1]
    
    # importances = lr.coef_[0]
    
    # test prediction
    y_test_pred = mnb.predict(X_test)
    y_test_proba = mnb.predict_proba(X_test)[:, 1]
    
    #print accuracy
    print_model_evaluation(model_name,"Train",metrics.accuracy_score(y_train, y_train_pred),y_train, y_train_pred, y_train_proba, y_pred_all, y_proba_all, save_data,
                           [dt.datetime.now(), description1, description2, "train", model_name,model_description,time_elapsed])
    print_model_evaluation(model_name,"Test",metrics.accuracy_score(y_test, y_test_pred),y_test, y_test_pred, y_test_proba,y_pred_all,y_proba_all,save_data,
                           [dt.datetime.now(),description1, description2, "test", model_name,model_description,time_elapsed])


def CallAllMLs (X_train, X_test, y_train, y_test, description1, description2):
  RandomForestModel_TrainTest(X_train, X_test, y_train, y_test, description1, description2)
  lightgbm_TrainTest(X_train, X_test, y_train, y_test, description1, description2)
  xgboost_TrainTest(X_train, X_test, y_train, y_test, description1, description2)
  LogisticRegression_TrainTest(X_train, X_test, y_train, y_test, description1, description2)

#%%


description1 = "no data augmentation, training size 0.20. 40 symp"
description2 = "1st run"
CallAllMLs (X_train, X_test, y_train, y_test, description1, description2)


#%%

def plot_all_ROC_in_one (y,y_proba,model):
  fpr = dict()
  tpr = dict()
  roc_auc = dict()
  fpr, tpr, _ = roc_curve(y, y_proba)
  roc_auc = np.round(auc(fpr, tpr),3)
  lw = 2
  plt.plot(fpr, tpr, lw=lw, label=f"{model}-{roc_auc}")


plt.figure(figsize=(8,8))
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')

for model_ in y_proba_all:
  if "Train" in model_: 
    
    y = y_train
  else : 
    continue
    y = y_test
  
  model = model_.replace("Train","").replace("Test","")
  plot_all_ROC_in_one (y,y_proba_all[model_],model)

plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title(f'All_Model Receiver operating characteristic')
plt.legend(loc="lower right")
plt.show()

time.sleep(5)

plt.figure(figsize=(8,8))
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')

for model_ in y_proba_all:
  if "Test" in model_: 
    y = y_test
  else : 
    continue
  
  model = model_.replace("Train","").replace("Test","")
  plot_all_ROC_in_one (y,y_proba_all[model_],model)

plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title(f'All_Model Receiver operating characteristic')
plt.legend(loc="lower right")
plt.show()
