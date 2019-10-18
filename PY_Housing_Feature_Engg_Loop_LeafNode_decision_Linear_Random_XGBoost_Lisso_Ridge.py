#!/usr/bin/env python
# coding: utf-8

# In[110]:


import numpy as np
import matplotlib.pyplot as plot
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
from numpy import loadtxt
import xgboost
import math
from sklearn.metrics import accuracy_score
from sklearn.metrics import explained_variance_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from math import sqrt
from sklearn import metrics


# In[111]:


## Read the housing.csv file 
data = pd.read_csv("housing.csv")


# In[112]:


## printed few rows from the housing file
data.head()


# In[113]:


## there are 20640 rows and 10 columns in the dataset
data.shape


# In[114]:


## in the total_bedrooms we have 207 rows which are null
data.isna().sum()


# In[115]:


## handling the null value of total bedroom and filling that with the mean value
df=pd.DataFrame(data)
data = df.fillna(int(df['total_bedrooms'].mean()))


# In[116]:


## Look into the unique values for ocean_proximity
data['ocean_proximity'].unique()


# In[117]:


## Converted the categorical column (ocean_proximity) in the dataset to numerical data
lb = LabelEncoder()
data['ocean_proximity'] = lb.fit_transform(data['ocean_proximity'])


# In[118]:


## Calculating the correlation of the data
data.corr()


# In[119]:


## We can visualize the correlation using the heat map.
plot.figure(figsize=(12,10))
cor = data.corr()
sns.heatmap(cor, annot=True, cmap=plot.cm.Reds)
plot.show()


# In[120]:


#Correlation with output variable
cor_target = abs(cor["median_house_value"])
#Selecting highly correlated features
relevant_features = cor_target[cor_target>0.5]
print(relevant_features)


# In[121]:


sns.boxplot(data=data['median_house_value'])


# In[122]:


sns.distplot(data['median_house_value'])


# In[123]:


## WE CAN SEE THAT THERE IS BINOMIAL DISTRIBUTION SO WE ARE ELIMINATING THE MEDIAN_HOUSE_VALUE > 500000 

data = data[data['median_house_value'] <= 500000]


# In[124]:


## We are defining the Dependent and Indepenedent variable and splitting the data into Test and Train 
x = data[['longitude','latitude','housing_median_age','total_rooms','total_bedrooms','population','households','median_income','ocean_proximity']]
y = data[['median_house_value']]
x_train, x_test, y_train, y_test =train_test_split(x,y, test_size = 0.20)


# # Linear Regression

# In[125]:


from sklearn.linear_model import LinearRegression
lm = LinearRegression()

### MODELFITSTRAIN - use the LinearRegression() function on my training data 
lm.fit(x_train,y_train)


# In[126]:


pred_y = lm.predict(x_test)


# In[127]:


pred_y_scatter = pd.DataFrame(pred_y).iloc[:,0]
y_test_scatter = pd.DataFrame(y_test).iloc[:,0]


# In[128]:


### Visualtion of predictive results, y_test for prediction 
sns.regplot(y_test_scatter,pred_y_scatter)


# In[129]:


sns.distplot((y_test-pred_y))


# In[130]:


## calculating the RMSE
rmse_linear = sqrt(mean_squared_error(y_test,pred_y))
print('RMSE : ',rmse_linear)


# In[131]:


#R Squared calculated.
r2_linear =r2_score(y_test,pred_y)
r2_linear


# # XG Boost ######################

# In[132]:


# split data into train and test sets
X_train_XG, X_test_XG, y_train_XG, y_test_XG = train_test_split(x, y, test_size=0.15, random_state=7)


# In[133]:


# Let's try XGboost algorithm to see if we can get better results
xgb = xgboost.XGBRegressor(n_estimators=100, learning_rate=0.40, gamma=0, subsample=0.75,
                           colsample_bytree=1, max_depth=7)


# In[134]:


xgb.fit(X_train_XG,y_train_XG)


# In[135]:


predictions_XG = xgb.predict(X_test_XG)

## Calculating the RMSE
rmse_XG = sqrt(mean_squared_error(predictions_XG,y_test_XG))
print('RMSE: ',rmse_XG)


# In[136]:


predictions_scatter_XG = pd.DataFrame(predictions_XG).iloc[:,0]
y_scatter_test_XG = pd.DataFrame(y_test_XG).iloc[:,0]


# In[137]:


### Visualtion of predictive results, y_test is actual y, pred_y is the predicted
sns.regplot(x=y_scatter_test_XG,y=predictions_scatter_XG)


# In[138]:


#RSquared calculation 
r2_XGBoost = r2_score(predictions_XG,y_test_XG)
r2_XGBoost


# # Decision Tree  ##################

# In[139]:


def get_md(leaf_nodes,depth,X_train_DT, X_test_DT, y_train_DT, y_test_DT):
    model = DecisionTreeRegressor(max_leaf_nodes=leaf_nodes,max_depth=depth,random_state=0)
    model.fit(X_train_DT,y_train_DT)
    a = model.predict(X_test_DT)
    mae = sqrt(mean_squared_error(a,y_test_DT))
    #mae = mean_absolute_error(y_test_DT,a)
    return(mae)


# In[140]:


from sklearn.tree import DecisionTreeRegressor
from sklearn import tree

X_train_DT, X_test_DT, y_train_DT, y_test_DT = train_test_split(x, y, test_size = 0.2) ##, random_state = 7
regressor = DecisionTreeRegressor(random_state = 0,max_leaf_nodes = 505,max_depth=9)
regressor.fit(X_train_DT, y_train_DT)


# In[141]:


for leaf_nodes in np.arange(5,5000,500):
    for depth in np.arange(1,10):
        mae = get_md(leaf_nodes,depth,X_train_DT, X_test_DT, y_train_DT, y_test_DT)
        print('leaf nodes',leaf_nodes,"  >>>>>>>>>>  ",'depth',depth,"  >>>>>>>>>>  ",mae)


# In[142]:


y_pred_DT = regressor.predict(X_test_DT)
y_pred_DT


# In[143]:


y_scatter_pred_DT = pd.DataFrame(y_pred_DT).iloc[:,0]
y_scatter_test_DT = pd.DataFrame(y_test_DT).iloc[:,0]


# In[144]:


### Visualtion of predictive results, y_test is actual y, pred_y is the predicted 
sns.regplot(y_scatter_test_DT,y_scatter_pred_DT)


# In[145]:


## Calculate the RMSE
rmse_DT = sqrt(mean_squared_error(y_pred_DT,y_test_DT))
print('RMSE: ',rmse_DT)


# In[146]:


##RSquared calculation
R2_DT = r2_score(y_pred_DT,y_test_DT)
R2_DT


# # Random Forest ###########

# In[147]:


# Using Skicit-learn to split data into training and testing sets
from sklearn.model_selection import train_test_split
# Split the data into training and testing sets
train_features, test_features, train_labels, test_labels = train_test_split(x, y, test_size = 0.20, random_state = 7)

# Import the model we are using
from sklearn.ensemble import RandomForestRegressor
# Instantiate model with 1000 decision trees
rf = RandomForestRegressor(n_estimators = 120, random_state = 0)
# Train the model on training data
rf.fit(train_features, train_labels);


# In[148]:


# Use the forest's predict method on the test data
predictions = rf.predict(test_features)


# In[149]:


rmse_RF = sqrt(mean_squared_error(predictions,test_labels))
print('RMSE :',rmse_RF)


# In[150]:


## Calculating the RMSE
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(test_labels, predictions)))


# In[151]:


predictions_scatter_RF = pd.DataFrame(predictions).iloc[:,0]
test_labels_scatter_RF = pd.DataFrame(test_labels).iloc[:,0]


# In[152]:


### Visualtion of predictive results, y_test is actual y, pred_y is the predicted 
sns.regplot(test_labels_scatter_RF,predictions_scatter_RF)


# In[153]:


#RSquared calculation
r2_RF = r2_score(test_labels, predictions)
r2_RF


# # Linear Regression with median Income ############################

# In[154]:


# will take only median_income as an variable
X = data[['median_income']]
Y = data[['median_house_value']]
X_train, X_test, Y_train, Y_test =train_test_split(X,Y, test_size = 0.20)


# In[155]:


from sklearn.linear_model import LinearRegression
lm = LinearRegression()

### use the LinearRegression() function on my training data 
lm.fit(X_train,Y_train)


# In[156]:


pred_Y = lm.predict(X_test)


# In[157]:


#pred_y_scatter = list(pred_y)
#y_test_scatter =list(y_test['median_house_value'])
pred_Y_scatter = pd.DataFrame(pred_Y).iloc[:,0]
Y_test_scatter = pd.DataFrame(Y_test).iloc[:,0]


# In[158]:


### Visualtion of predictive results, y_test is actual y, pred_y is the predicted 
sns.regplot(Y_test_scatter,pred_Y_scatter)


# In[159]:


## Calculate the RMSE
RMSE = sqrt(mean_squared_error(Y_test,pred_Y))
print('RMSE: ',RMSE)


# In[160]:


#RSquared calculation
r2_score(Y_test,pred_Y)


# # XG BOOST with median Income ############################

# In[161]:


# split data into train and test sets
x_train_XG, x_test_XG, Y_train_XG, Y_test_XG = train_test_split(X, Y, test_size=0.20, random_state=7)


# In[162]:


# Let's try XGboost algorithm to see if we can get better results
xgb = xgboost.XGBRegressor(n_estimators=100, learning_rate=0.25, gamma=0, subsample=0.75,
                           colsample_bytree=1, max_depth=7)


# In[163]:


xgb.fit(x_train_XG,Y_train_XG)


# In[164]:


Predictions_XG = xgb.predict(x_test_XG)
RMSE_XG = sqrt(mean_squared_error(Predictions_XG,Y_test_XG))
print('RMSE: ',RMSE_XG)


# In[165]:


#pred_y_scatter = list(pred_y)
#y_test_scatter =list(y_test['median_house_value'])
Predictions_scatter_XG = pd.DataFrame(Predictions_XG).iloc[:,0]
Y_scatter_test_XG = pd.DataFrame(Y_test_XG).iloc[:,0]


# In[166]:


### Visualtion of predictive results, y_test is actual y, pred_y is the predicted 
sns.regplot(Y_scatter_test_XG,Predictions_scatter_XG)


# In[167]:


#RSquared calculation
r2_score(Y_scatter_test_XG,Predictions_scatter_XG)


# # Comparing RMSE and R Squared for all models ########################

# In[168]:


## Calculation of RMSE and compare with all the model that we have implemented.
rmse_compare = [{'Linear':rmse_linear,'Decision Tree':rmse_DT,'Random Forest':rmse_RF,'XG Boost':rmse_XG}]
rmse_compare = pd.DataFrame(rmse_compare)
sns.barplot(data=rmse_compare)


# From the above graph we can predict that the Random Forest and XG Boost algorithm gives us the least 'Sqrt Root mean square'

# In[169]:


## Calculation of R Squared and compare with all the model that we have implemented.
rsquared_compare = [{'Linear':r2_linear,'Decision Tree':R2_DT,'Random Forest':r2_RF,'XG Boost':r2_XGBoost}]
rsquared_compare = pd.DataFrame(rsquared_compare)
sns.barplot(data=rsquared_compare)


# From the above graph we can predict that the Random Forest and XG Boost algorithm gives us the high R Squared value

# ## We can do the feature engineering on the above model as there is lot of scope of improvement.

# ## 1) SelectKBest

# In[170]:


from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
test = SelectKBest(score_func= chi2, k=4)


# In[171]:


x_f = data[['longitude','latitude','housing_median_age','total_rooms','total_bedrooms','population','households','median_income','ocean_proximity']]
y_f = data[['median_house_value']]
x_f['longitude_abs'] = x_f['longitude'].abs()
x_f=x_f.reindex(columns=['longitude_abs','latitude','housing_median_age','total_rooms','total_bedrooms','population','households','median_income','ocean_proximity'])
fit = test.fit(x_f,y_f)


# In[172]:


np.set_printoptions(precision=3)
print(fit.scores_)


# In[173]:


features = fit.transform(x_f)
# summarize selected features
print(features[0:,:])


# In[174]:


# Using Skicit-learn to split data into training and testing sets

x_f_skb = x_f[['total_rooms','total_bedrooms','population','households']]
y_f_skb = y_f[['median_house_value']]

from sklearn.model_selection import train_test_split
# Split the data into training and testing sets
train_features, test_features, train_labels, test_labels = train_test_split(x_f_skb, y_f_skb, test_size = 0.30, random_state = 7)

# Import the model we are using
from sklearn.ensemble import RandomForestRegressor
# Instantiate model with 1000 decision trees
rf = RandomForestRegressor(n_estimators = 120, random_state = 0)
# Train the model on training data
rf.fit(train_features, train_labels);

# Use the forest's predict method on the test data
predictions = rf.predict(test_features)

#RSquared calculation
r2_RF = r2_score(test_labels, predictions)
r2_RF

From the above SelectKBest am getting 'total_rooms','total_bedrooms','population','households' as 4 best feature but we are not getting the expected R Squared. So we can't take the SelectKBest features.
# ## 2) RFE

# In[175]:


from sklearn.feature_selection import RFE
model = RandomForestRegressor()
rfe = RFE(model, 4)
fit = rfe.fit(x, y)
print("Num Features: %d" % fit.n_features_)
print("Selected Features: %s" % fit.support_)
print("Feature Ranking: %s" % fit.ranking_)

##['longitude_abs','latitude','housing_median_age','total_rooms','total_bedrooms','population','households','median_income','ocean_proximity']
## from the above RME we can find that the 'longitude_abs','latitude','median_income','ocean_proximity'
# In[176]:


# Using Skicit-learn to split data into training and testing sets

x_f_skb = x_f[['longitude_abs','latitude','population','households','housing_median_age']]
y_f_skb = y_f[['median_house_value']]

from sklearn.model_selection import train_test_split
# Split the data into training and testing sets
train_features, test_features, train_labels, test_labels = train_test_split(x_f_skb, y_f_skb, test_size = 0.30, random_state = 7)

# Import the model we are using
from sklearn.ensemble import RandomForestRegressor
# Instantiate model with 1000 decision trees
rf = RandomForestRegressor(n_estimators = 120, random_state = 0)
# Train the model on training data
rf.fit(train_features, train_labels);

# Use the forest's predict method on the test data
predictions = rf.predict(test_features)

#RSquared calculation
r2_RF = r2_score(test_labels, predictions)
r2_RF


# ## Backward Elimination OLS Method

# In[177]:


import statsmodels.api as sm
#Adding constant column of ones, mandatory for sm.OLS model
X_1 = sm.add_constant(x_f)
#Fitting sm.OLS model
model = sm.OLS(y_f,X_1).fit()
model.pvalues


# In[178]:


#Backward Elimination
cols = list(x_f.columns)
pmax = 1
while (len(cols)>0):
    p= []
    X_1 = x_f[cols]
    X_1 = sm.add_constant(X_1)
    model = sm.OLS(y_f,X_1).fit()
    p = pd.Series(model.pvalues.values[1:],index = cols)      
    pmax = max(p)
    feature_with_p_max = p.idxmax()
    if(pmax>0.05):
        cols.remove(feature_with_p_max)
    else:
        break
selected_features_BE = cols
print(selected_features_BE)


# In[179]:


# Using Skicit-learn to split data into training and testing sets

x_f_skb = x_f[['longitude_abs', 'latitude', 'housing_median_age', 'total_rooms', 'total_bedrooms', 'population', 'households', 'median_income']]
y_f_skb = y_f[['median_house_value']]

from sklearn.model_selection import train_test_split
# Split the data into training and testing sets
train_features, test_features, train_labels, test_labels = train_test_split(x_f_skb, y_f_skb, test_size = 0.30, random_state = 7)

# Import the model we are using
from sklearn.ensemble import RandomForestRegressor
# Instantiate model with 1000 decision trees
rf = RandomForestRegressor(n_estimators = 120, random_state = 0)
# Train the model on training data
rf.fit(train_features, train_labels);

# Use the forest's predict method on the test data
predictions = rf.predict(test_features)

#RSquared calculation
r2_RF = r2_score(test_labels, predictions)
r2_RF

## So via OLS we have eliminated ocean_proximity and getting good R2 value 
# ## Embedded Method

# In[180]:


from sklearn.linear_model import RidgeCV, LassoCV, Ridge, Lasso
reg = LassoCV()
reg.fit(x_f, y_f)
print("Best alpha using built-in LassoCV: %f" % reg.alpha_)
print("Best score using built-in LassoCV: %f" %reg.score(x_f,y_f))
coef = pd.Series(reg.coef_, index = x_f.columns)


# In[181]:


coef


# In[182]:



imp_coef = coef.sort_values()
import matplotlib
matplotlib.rcParams['figure.figsize'] = (8.0, 10.0)
imp_coef.plot(kind = "barh")
plot.title("Feature importance using Lasso Model")


# In[183]:


## using the Lasso it says that we have to remove Ocean Proximity, Latitude, Longitude_abs total_rooms so lets check after removing how much we are getting the RSquared. 


# In[184]:


# Using Skicit-learn to split data into training and testing sets

x_f_skb = x_f[['housing_median_age', 'total_bedrooms', 'population', 'households', 'median_income', 'total_rooms']]
y_f_skb = y_f[['median_house_value']]

from sklearn.model_selection import train_test_split
# Split the data into training and testing sets
train_features, test_features, train_labels, test_labels = train_test_split(x_f_skb, y_f_skb, test_size = 0.30, random_state = 7)

# Import the model we are using
from sklearn.ensemble import RandomForestRegressor
# Instantiate model with 1000 decision trees
rf = RandomForestRegressor(n_estimators = 120, random_state = 0)
# Train the model on training data
rf.fit(train_features, train_labels);

# Use the forest's predict method on the test data
predictions = rf.predict(test_features)

#RSquared calculation
r2_RF = r2_score(test_labels, predictions)
r2_RF

## so after the lasso we can see that the R Squared is 60...So after the analysis we found out that we can consider OLS method for feature engineering.## Also lets check do we have any multicollinearity in the selected feature.
# In[185]:


data_mc = data[['longitude','latitude','housing_median_age','total_rooms','total_bedrooms','population',
               'households','median_income','median_house_value']]


# In[186]:


data_mc.corr()


# In[187]:


data_mc1 = data[['longitude','latitude']]
data_mc1.corr()


# ##Lasso

# In[188]:


X_train, X_test , y_train, y_test = train_test_split(x, y, test_size=0.5, random_state=1)


# In[189]:


def get_lasso(alpha_input,X_train, X_test , y_train, y_test):
    ridge = Ridge(alpha = alpha_input, normalize = True)
    ridge.fit(X_train, y_train)
    pred_lasso = ridge.predict(X_test)
    mae = sqrt(mean_squared_error(pred_lasso,y_test))
    return(mae) 

#    model = DecisionTreeRegressor(max_leaf_nodes=leaf_nodes,max_depth=depth,random_state=0)
#    model.fit(X_train_DT,y_train_DT)
#    a = model.predict(X_test_DT)
#    mae = sqrt(mean_squared_error(a,y_test_DT))
#    #mae = mean_absolute_error(y_test_DT,a)
#    return(mae)


# In[201]:


a=[0.0000001,0.00001,0.0001,0.001,0.01,0.1,1,2,3,4,5]
for alpha_input in a:
    mae = get_lasso(alpha_input,X_train, X_test , y_train, y_test)
    print('alpha ',alpha_input,'>>>>>',mae)


# In[202]:


from sklearn.linear_model import Ridge, RidgeCV, Lasso, LassoCV
ridge = Ridge(alpha = 1e-05, normalize = True)
ridge.fit(X_train, y_train)             # Fit a ridge regression on the training data
pred_lasso = ridge.predict(X_test)           # Use this model to predict the test data
sqrt(mean_squared_error(pred_lasso,y_test))


# In[192]:


ridge.coef_


# In[204]:


reg = LassoCV(max_iter = 10)
reg.fit(X_train, y_train)
print("Best alpha using built-in LassoCV: %f" % reg.alpha_)


# In[196]:


print(lm.coef_)


# So above were various method where we can find the best feature which we can use in our model.

# In[ ]:




