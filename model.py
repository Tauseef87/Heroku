import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle

dataset = pd.read_csv('hiring.csv')
dataset['experience'].fillna(0,inplace=True)
dataset['test_score'].fillna(0,inplace=True)

X= dataset.iloc[:,:3]

#Convert word to integer

def con_to_int(word):
    word_dict = {'one':1,'two':2,'three':3,'Four':4,'five':5,'Six':6,
                 'seven':7,'Eight':8,'Nine':9,'ten':10,'eleven':11
                 ,'twelve':12,'Zero':0,0:0}
    return word_dict[word]

X['experience']=X['experience'].apply(lambda x:con_to_int(x))

y = dataset.iloc[:,-1]

from sklearn.linear_model import LinearRegression
reg = LinearRegression()

reg.fit(X,y)

#Saving model to disk

pickle.dump(reg,open('model.pkl','wb'))

#Loading model to compare result
model = pickle.load(open('model.pkl','rb'))
print(model.predict([[2,9,6]]))
