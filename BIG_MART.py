# -*- coding: utf-8 -*-
"""
Created on Mon Jun 26 08:52:04 2017

@author: Alex
"""

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn import metrics
from sklearn.ensemble import RandomForestRegressor
import keras
from keras.models import Model,Sequential
from keras.callbacks import TensorBoard
from keras.layers import Input, Dense, Reshape, Dropout, Activation
from keras.optimizers import SGD, Adam


### Importing the Data
train_set = pd.read_csv("E:/BDAP/Datasets/AV Meetup Problem/BIG_MART_Trainset.csv",header=0)
test_set = pd.read_csv("E:/BDAP/Datasets/AV Meetup Problem/BIG_MART_Testset.csv",header=0)
train_set.head()
train_set.describe()

### checking for the correlation
sns.heatmap(train_set.corr())

### handling the Train data
train_set.columns
pd.Categorical(train_set['Outlet_Size']).describe()
train_set['Outlet_Size'] = train_set['Outlet_Size'].fillna('a',limit=565)
train_set['Outlet_Size'] = train_set['Outlet_Size'].replace('a','High')
train_set['Outlet_Size'] = train_set['Outlet_Size'].fillna('a',limit=923)
train_set['Outlet_Size'] = train_set['Outlet_Size'].replace('a','Medium')
train_set['Outlet_Size'] = train_set['Outlet_Size'].fillna('a')
train_set['Outlet_Size'] = train_set['Outlet_Size'].replace('a','Small')
pd.Categorical(train_set['Outlet_Size']).describe()
pd.Categorical(train_set['Item_Fat_Content']).describe()
train_set['Item_Fat_Content'] = train_set['Item_Fat_Content'].replace(('low fat','LF'),'Low Fat')
train_set['Item_Fat_Content'] = train_set['Item_Fat_Content'].replace('reg','Regular')
sum(np.isnan(train_set['Item_Weight']))   ### counting the NA
train_set['Item_Weight'] = train_set.Item_Weight.fillna(train_set.Item_Weight.mean())
train_set['Item_Visibility'] = train_set.Item_Visibility.replace(0,train_set.Item_Visibility.mean())
plt.hist(train_set['Item_Visibility'])

train_set['Outlet_Size'] = train_set['Outlet_Size'].replace('High', 0)      
train_set['Outlet_Size'] = train_set['Outlet_Size'].replace('Medium',1)         
train_set['Outlet_Size'] = train_set['Outlet_Size'].replace('Small',2)
train_set['Item_Fat_Content'] = train_set['Item_Fat_Content'].replace('Low Fat', 0)
train_set['Item_Fat_Content'] = train_set['Item_Fat_Content'].replace('Regular',1)
train_set['Outlet_Establishment_Year'] = 2013 - train_set['Outlet_Establishment_Year']
train_set['Outlet_Type'] = train_set['Outlet_Type'].replace('Grocery Store', 0)
train_set['Outlet_Type'] = train_set['Outlet_Type'].replace('Supermarket Type1', 0)
train_set['Outlet_Type'] = train_set['Outlet_Type'].replace('Supermarket Type2', 1)
train_set['Outlet_Type'] = train_set['Outlet_Type'].replace('Supermarket Type3', 2)
train_set['Outlet_Location_Type'] = train_set['Outlet_Location_Type'].replace('Tier 1', 0)
train_set['Outlet_Location_Type'] = train_set['Outlet_Location_Type'].replace('Tier 2', 1)
train_set['Outlet_Location_Type'] = train_set['Outlet_Location_Type'].replace('Tier 3', 2)
y = train_set["Item_Outlet_Sales"]
x = pd.get_dummies(train_set['Item_Type'])
train_set = pd.concat([train_set.iloc[:,1:],x],axis=1)
train_set = train_set.drop(['Item_Type'],axis=1)
x = pd.get_dummies(train_set['Outlet_Identifier'])
train_set = pd.concat([train_set,x],axis=1)
train_set = train_set.drop(['Outlet_Identifier'],axis=1)
train_set = train_set.drop(['Item_Outlet_Sales'],axis=1)
train_set.dtypes




#### Divinding the Train set into train and cross validation set

x_train, x_cv, y_train, y_cv = train_test_split(train_set,y)
y_cv

##### Fitting a regression model
lm = LinearRegression()
lm.fit(x_train,y_train)
coeff_df = pd.DataFrame(lm.coef_,train_set.columns,columns=['Coefficient'])
coeff_df
lm.residues_
predict = lm.predict(x_cv)
predict
lm.score(x_cv,y_cv)
rmse1 = np.sqrt(sum((predict-y_cv)**2)/len(y_cv))


#######################################################
### Ridge

ridge = Ridge(alpha= 0.01,normalize=True)
ridge.fit(x_train,y_train)
coeff_df = pd.DataFrame(ridge.coef_,train_set.columns,columns=['Coefficient'])
coeff_df
predict = ridge.predict(x_cv)
ridge.score(x_cv,y_cv)
rmse2 = np.sqrt(sum((predict-y_cv)**2)/len(y_cv))

### Lasso
lasso = Lasso(alpha=0.1, normalize=True)
lasso.fit(x_train,y_train)
coeff_df = pd.DataFrame(lasso.coef_,train_set.columns,columns=['Coefficient'])
coeff_df
predict = lasso.predict(x_cv)
lasso.score(x_cv,y_cv)
rmse3 = np.sqrt(sum((predict-y_cv)**2)/len(y_cv))

### Elastic Net
elastic = ElasticNet(alpha=0.01, l1_ratio=1, normalize=False)
elastic.fit(x_train,y_train)
coeff_df = pd.DataFrame(elastic.coef_,train_set.columns,columns=['Coefficient'])
coeff_df
predict = elastic.predict(x_cv)
elastic.score(x_cv,y_cv)
rmse4 = np.sqrt(metrics.mean_squared_error(y_cv,predict))

### Random Forest
lrf = RandomForestRegressor(n_estimators=500,min_samples_split=350)
lrf.fit(x_train,y_train)
predict = lrf.predict(x_cv)
rmse5 = np.sqrt(metrics.mean_squared_error(y_cv,predict))
rmse5

### Neural network
x_train = x_train.values
y_train =y_train.values
x_cv = x_cv.values
y_cv = y_cv.values

lr = 0.001 ##### learning rate
wd = 0.01
batch_size = 128
num_epochs = 50
model = Sequential()
model.add(Dense(512,input_dim=34,activation='relu',kernel_initializer='normal'))
model.add(Dense(256,input_dim=34, activation = 'relu',kernel_initializer='normal'))
model.add(Dense(512,input_dim=34, activation = 'relu',kernel_initializer='normal'))
model.add(Dense(256,input_dim=34, activation = 'relu',kernel_initializer='normal'))
model.add(Dense(780,input_dim=34,activation='relu',kernel_initializer='normal'))
model.add(Dense(220,input_dim=34,activation='relu',kernel_initializer='normal'))
model.add(Dense(1, kernel_initializer='normal'))
#sgd = SGD(lr=lr, decay=1e-3, momentum=0.9, nesterov=True, clipvalue=5.0)
adam = Adam(lr=lr, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
model.compile(loss='mean_squared_error', optimizer=adam,metrics=['accuracy'])
print (model.summary())
model.fit(x_train, y_train, 
    nb_epoch=num_epochs, 
    batch_size=batch_size, 
    shuffle=True, 
    validation_data=(x_cv, y_cv))

predicts = model.predict(x_cv)
predicts = pd.DataFrame(predicts)
rmse6 = np.sqrt(metrics.mean_squared_error(y_cv,predicts))
rmse6


################# Test Data #################

test_set.head()
test_set.describe()

test_set.columns
pd.Categorical(test_set['Outlet_Size']).describe()
test_set['Outlet_Size'] = test_set['Outlet_Size'].fillna('a',limit=300)
test_set['Outlet_Size'] = test_set['Outlet_Size'].replace('a','High')
test_set['Outlet_Size'] = test_set['Outlet_Size'].fillna('a',limit=800)
test_set['Outlet_Size'] = test_set['Outlet_Size'].replace('a','Medium')
test_set['Outlet_Size'] = test_set['Outlet_Size'].fillna('a')
test_set['Outlet_Size'] = test_set['Outlet_Size'].replace('a','Small')
pd.Categorical(test_set['Outlet_Size']).describe()
pd.Categorical(test_set['Item_Fat_Content']).describe()
test_set['Item_Fat_Content'] = test_set['Item_Fat_Content'].replace(('low fat','LF'),'Low Fat')
test_set['Item_Fat_Content'] = test_set['Item_Fat_Content'].replace('reg','Regular')
sum(np.isnan(test_set['Item_Weight']))   ### counting the NA
test_set['Item_Weight'] = test_set.Item_Weight.fillna(test_set.Item_Weight.mean())
test_set['Item_Visibility'] = test_set.Item_Visibility.replace(0,test_set.Item_Visibility.mean())
plt.hist(train_set['Item_Visibility'])

test_set['Outlet_Size'] = test_set['Outlet_Size'].replace('High', 0)      
test_set['Outlet_Size'] = test_set['Outlet_Size'].replace('Medium',1)         
test_set['Outlet_Size'] = test_set['Outlet_Size'].replace('Small',2)
test_set['Item_Fat_Content'] = test_set['Item_Fat_Content'].replace('Low Fat', 0)
test_set['Item_Fat_Content'] = test_set['Item_Fat_Content'].replace('Regular',1)
test_set['Outlet_Establishment_Year'] = 2013 - test_set['Outlet_Establishment_Year']
test_set['Outlet_Type'] = test_set['Outlet_Type'].replace('Grocery Store', 0)
test_set['Outlet_Type'] = test_set['Outlet_Type'].replace('Supermarket Type1', 0)
test_set['Outlet_Type'] = test_set['Outlet_Type'].replace('Supermarket Type2', 1)
test_set['Outlet_Type'] = test_set['Outlet_Type'].replace('Supermarket Type3', 2)
test_set['Outlet_Location_Type'] = test_set['Outlet_Location_Type'].replace('Tier 1', 0)
test_set['Outlet_Location_Type'] = test_set['Outlet_Location_Type'].replace('Tier 2', 1)
test_set['Outlet_Location_Type'] = test_set['Outlet_Location_Type'].replace('Tier 3', 2)
x = pd.get_dummies(test_set['Item_Type'])
test_set = pd.concat([test_set.iloc[:,1:],x],axis=1)
test_set = test_set.drop(['Item_Type'],axis=1)
x = pd.get_dummies(test_set['Outlet_Identifier'])
test_set = pd.concat([test_set,x],axis=1)
test_set = test_set.drop(['Outlet_Identifier'],axis=1)
test_set.dtypes


##### Fitting a regression model


lm.fit(train_set,y)
coeff_df = pd.DataFrame(lm.coef_,train_set.columns,columns=['Coefficient'])
coeff_df
lm.residues_
predict = lm.predict(test_set)
predict
model1 = pd.DataFrame(predict,columns=['Linear predict'])

ridge.fit(train_set,y)
coeff_df = pd.DataFrame(ridge.coef_,train_set.columns,columns=['Coefficient'])
coeff_df
predict = ridge.predict(test_set)
predict
model2 = pd.DataFrame(predict,columns=['Ridge predict'])


lasso.fit(train_set,y)
coeff_df = pd.DataFrame(lasso.coef_,train_set.columns,columns=['Coefficient'])
coeff_df
predict = lasso.predict(test_set)
predict
model3 = pd.DataFrame(predict,columns=['Lasso predict'])


elastic.fit(train_set,y)
coeff_df = pd.DataFrame(elastic.coef_,train_set.columns,columns=['Coefficient'])
coeff_df
predict = elastic.predict(test_set)
predict
model4 = pd.DataFrame(predict,columns=['Elastic predict'])

lrf.fit(train_set,y)
predict = lrf.predict(test_set)
model5 = pd.DataFrame(predict,columns=['RandomForest predict'])

X = train_set.values
Y = y.values
test = test_set.values
model.fit(X,Y, 
    nb_epoch=num_epochs, 
    batch_size=batch_size, 
    shuffle=True)

predicts = model.predict(test)
model6 = pd.DataFrame(predicts,columns=['NNetwork predict'])

test_final = pd.concat((model2,model1,model3,model4),axis=1)
test_final.to_csv("E:/BDAP/Datasets/AV Meetup Problem/file1.csv")





##Outlet_Identifier - OUT010, OUT019, OUT018, OUT013 which has lowest sales