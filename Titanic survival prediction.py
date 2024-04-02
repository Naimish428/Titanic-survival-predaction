#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np 
import pandas as pd


# In[4]:


df = pd.read_csv("D:/codsoft/Titanic-Dataset.csv")


# In[6]:


df.head


# In[8]:


df.shape


# In[10]:


df.info()


# In[12]:


df.isnull().sum()


# In[14]:


Survived = df[df["Survived"]==1]
Non_Survived = df[df["Survived"]==0]
outlier = len(Survived)/float(len(Non_Survived))
print(outlier)
print("Survived : {} " .format(len(Survived)))
print("Non_Survived : {} " .format(len(Non_Survived)))


# In[16]:


import seaborn as sns
sns.countplot(x= df["Survived"] , hue = df["Pclass"])


# In[18]:


import sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


# In[20]:


labelencoder = LabelEncoder() # Conversion of Categorical values into Numerical values
df['Sex'] = labelencoder.fit_transform(df['Sex'])

df.head()


# In[24]:


features = df[["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare"]]
target = df["Survived"]


# In[26]:


df['Age'].fillna(df['Age'].median(), inplace=True)


# In[28]:


df.isnull().sum()


# In[30]:


x=df[['Pclass', 'Sex']]
y=target


# In[32]:


x_train , x_test, y_train, y_test = train_test_split(features,y, test_size= 0.2, random_state= 0)

from sklearn.impute import SimpleImputer # It is used to fill the missing values of the dataset.
imputer = SimpleImputer(strategy='mean')
x_train_imputed = imputer.fit_transform(x_train)
x_test_imputed = imputer.transform(x_test)


# In[34]:


model = RandomForestClassifier()
model.fit(x_train_imputed, y_train)


# In[36]:


predictions = model.predict(x_test_imputed)


# In[38]:


from sklearn.metrics import accuracy_score, precision_score , recall_score , f1_score
acc = accuracy_score(y_test , predictions)
print("The accuracy is {}".format(acc))

prec = precision_score(y_test , predictions)
print("The precision is {}".format(prec))

rec = recall_score(y_test , predictions)
print("The recall is {}".format(rec))

f1 = f1_score(y_test , predictions)
print("The F1-Score is {}".format(f1))


# In[40]:


import joblib
joblib.dump(model,"Titanic_Survival")


# In[42]:


m = joblib.load("Titanic_Survival")


# In[44]:


prediction  = m.predict([[1,1,0,1,1,1]])
prediction


# In[45]:


if prediction==0:
  print("Non Survived")
else:
  print("Survived")

