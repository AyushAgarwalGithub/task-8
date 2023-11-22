import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pickle

df=pd.read_csv("healthcare_dataset.csv")

df=df.drop(['Name','Doctor','Date of Admission','Discharge Date','Room Number','Hospital'],axis=1)

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
j= ['Gender','Blood Type','Medical Condition','Insurance Provider','Admission Type','Medication','Test Results']
for i in j:
  df[i]=le.fit_transform(df[i])

X=df.drop(['Test Results'],axis=1)
y=df['Test Results']

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(X_train)
x_test = sc.fit_transform(X_test)

from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
algo1=lr.fit(x_train,y_train)

pickle.dump(algo1,open("model.pkl","wb"))