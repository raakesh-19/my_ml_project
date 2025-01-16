import json as jd
import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn 
import xgboost
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn import metrics

file_path = "C:\\Users\\prana\\OneDrive\\Desktop\\2023.json"
with open(file_path,"rb") as file:
 data = jd.load(file)
 df = pd.read_json(file_path)
 print (df.columns)
 print (df.shape)
 df.describe()

sns.set()
plt.figure(figsize=(7,7))
sns.displot(df["item_per_rate"])
plt.show()

 
sns.set()
plt.figure(figsize=(6,6))
sns.displot(df["item_purchase_rate"])
plt.show()


sns.set()
plt.figure(figsize=(6,6))
sns.displot(df["item_sales_rate"])
plt.show()

sns.set()
plt.figure(figsize=(6,6))
sns.countplot(x="invoice_sales_person_name",data=df)
plt.show()

sns.set()
plt.figure(figsize=(6,600))
sns.barplot(df["customer_name"])
plt.show()


e=LabelEncoder()
df["customer_name"]=e.fit_transform(df["customer_name"])


e=LabelEncoder()
df["customer_id"]=e.fit_transform(df["customer_id"])


e=LabelEncoder()
df["item_sku"]=e.fit_transform(df["item_sku"])


e=LabelEncoder()
df["invoice_sales_person_name"]=e.fit_transform(df["invoice_sales_person_name"])


e=LabelEncoder()
df["item_id"]=e.fit_transform(df["item_id"])


e=LabelEncoder()
df['invoice_sales_person_id']=e.fit_transform(df['invoice_sales_person_id'])


e=LabelEncoder()
df["invoice_status"]=e.fit_transform(df["invoice_status"])


e=LabelEncoder()
df["invoice_currency"]=e.fit_transform(df["invoice_currency"])

e=LabelEncoder()
df["invoice_id"]=e.fit_transform(df["invoice_id"])



e=LabelEncoder()
df["item_line_id"]=e.fit_transform(df["item_line_id"])


e=LabelEncoder()
df["item_name"]=e.fit_transform(df["item_name"])



e=LabelEncoder()
df["invoice_date"]=e.fit_transform(df["invoice_date"])

e=LabelEncoder()
df["invoice_due_date"]=e.fit_transform(df["invoice_due_date"])

e=LabelEncoder()
df["invoice_number"]=e.fit_transform(df["invoice_number"])


X=df.drop(columns="item_sales_rate",axis=1)
Y=df["item_sales_rate"]

X_train,X_test,Y_train,Y_test= train_test_split(X,Y, test_size=0.2,random_state=2)

r = XGBRegressor()
r.fit(X_train, Y_train)

training_data_prediction = r.predict(X_train)
r2_train=metrics.r2_score(Y_train,training_data_prediction)

print("training R squared value",r2_train)

testing_data_prediction = r.predict(X_test)
r2_test=metrics.r2_score(Y_test,testing_data_prediction)

print("testing R squared value",r2_test)
