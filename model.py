#importing libs
import pandas as pd
import numpy as np
import pickle

df=pd.read_csv("SUV_Purchase.csv")


#feature engineering
df.drop('User ID', inplace=True, axis=1)
df.drop('Gender', inplace=True, axis=1)

#METHOD1
X=df.iloc[:,:-1].values  #2DARRAY
Y=df.iloc[:,-1:].values

#splitting the dataset
#spliting the data set into train and test dataset
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=0)

from sklearn.preprocessing import StandardScaler
sst=StandardScaler()

X_train = sst.fit_transform(X_train)
X_test = sst.transform(X_test)

#train model

from sklearn.linear_model import LogisticRegression
model=LogisticRegression()
model.fit(X_train,Y_train)

y_pred=model.predict(X_test)
print(y_pred)

pickle.dump(model,open('model.pkl','wb')) #serializing the model by creating "model.pkl
model=pickle.load(open('model.pkl','rb'))  #deserializing reading the file
print("success loaded")

#execte this file only once create a pickle file


