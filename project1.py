# project for online frauds
# for dataset access the below link
# https://drive.google.com/file/d/11qs186EvvN0MJoK90Bfv_iDqBXNzA48Y/view?usp=share_link

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data=pd.read_csv("C:\\Users\\DELL\\Downloads\\datasets\\online fraud.csv")
print(data.head())

print(data.isnull().sum())

print(data.type.value_counts())


type=data["type"]
transaction=type.value_counts().index.tolist()
quantities=type.value_counts().tolist()
plt.pie(x=quantities, labels=transaction, autopct='%1.1f%%')
plt.show()

correlation=data.corr()
print(correlation)

print(correlation["isFraud"].sort_values(ascending=False))

data["type"] = data["type"].map({"CASH_OUT": 1, "PAYMENT": 2, 
                                 "CASH_IN": 3, "TRANSFER": 4,
                                 "DEBIT": 5})
data["isFraud"] = data["isFraud"].map({0: "No Fraud", 1: "Fraud"})
print(data.head())

from sklearn.model_selection import train_test_split
y=np.array(data[["isFraud"]])
data.drop("isFraud",axis=1)

def correlation(dataset,threshold):
    col_corr=set()
    corr_matrix=dataset.corr()
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if (corr_matrix.iloc[i,j]) >threshold:
                colname=corr_matrix.columns[i]
                col_corr.add(colname)
    return col_corr


corr_feature=correlation(data,0.7)
print(corr_feature)
data.drop(corr_feature,axis=1)
x=np.array(data[["type","amount","oldbalanceOrg","oldbalanceDest"]])

from sklearn.tree import DecisionTreeClassifier
xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.10,random_state=42)
model=DecisionTreeClassifier()
model.fit(xtrain,ytrain)
print(model.score(xtest,ytest))

features = np.array([[4, 9000.60, 9000.60, 0.0]])
print(model.predict(features))