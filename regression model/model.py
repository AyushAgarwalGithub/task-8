import pandas as pd
import pickle

df=pd.read_csv("winequality-red.csv")
df.drop_duplicates(inplace=True)

X=df.drop(['quality'],axis=1)
y=df['quality']

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(X_train)
x_test = sc.fit_transform(X_test)

from sklearn.ensemble import RandomForestClassifier

rfc=RandomForestClassifier()
algo3=rfc.fit(x_train,y_train)

pickle.dump(algo3,open("model.pkl","wb"))