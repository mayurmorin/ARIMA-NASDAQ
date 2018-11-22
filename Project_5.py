
# coding: utf-8

# ## Pick up the following stocks and generate forecasts accordingly
# ### Stocks:
# <p>1. NASDAQ.AAPL</p>
# <p>2. NASDAQ.ADP</p>
# <p>3. NASDAQ.CBOE</p>
# <p>4. NASDAQ.CSCO</p>
# <p>5. NASDAQ.EBAY</p>

# <p>Dataset Link: https://drive.google.com/file/d/1VxoJDgyiAdMRI7-Fp7RxazDTvQ9Lw54d/ </p>

# ### Importing Modules

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas.tools.plotting import autocorrelation_plot
from statsmodels.graphics.tsaplots import plot_pacf
from statsmodels.tsa.arima_model import ARIMA, ARMAResults
import datetime
import sys
import seaborn as sns
import statsmodels
import statsmodels.stats.diagnostic as diag
from statsmodels.tsa.stattools import adfuller
from scipy.stats.mstats import normaltest
from matplotlib.pyplot import acorr
plt.style.use('fivethirtyeight')
get_ipython().run_line_magic('matplotlib', 'inline')
import warnings
warnings.filterwarnings('ignore')
import datetime as dt
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error


# ### Loading Data

# In[2]:


#Read CSV (comma-separated) file into DataFrame
df = pd.read_csv('data_stocks.csv')


# ### Data Exploration/Analysis

# In[3]:


#Returns the first 5 rows of df dataframe
df.head()


# In[4]:


df.describe() #The summary statistics of the df dataframe


# In[5]:


df.info() #Prints information about df DataFrame.


# In[6]:


df.columns #Columns of df dataframe


# In[7]:


df.shape #Return a tuple representing the dimensionality of df DataFrame.


# In[8]:


df.isnull().values.any() #Check for any NA’s in the dataframe.


# ## 1. NASDAQ.AAPL

# In[9]:


df_1 = df.copy() #Makes a copy of df dataframe.


# In[10]:


#Creating a column 'AAPL_LOG' with the log values of 'NASDAQ.AAPL' column data
df_1["AAPL_LOG"] = df_1["NASDAQ.AAPL"].apply(lambda x:np.log(x)) 


# In[11]:


df_1.head() #Returns the first 5 rows of df_1 dataframe


# In[12]:


type(df_1["DATE"][0]) #Type of the 'DATE' column


# In[13]:


#Creating a new column 'DATE_NEW' with formatted timestamp 
df_1["DATE_NEW"] = df_1["DATE"].apply(lambda x:dt.datetime.fromtimestamp(x).strftime("%Y-%m-%d %H:%M:%S"))


# In[14]:


df_1.head() #Returns the first 5 rows of df_1 dataframe


# In[15]:


#Positive serial correlation is associated with Durbin-Watson values below 2 and negative serial correlation with Durbin-Watson values above 2. 
#Prints Durbin-Watson statistic of given data.
print("Durbin-Watson statistic:",sm.stats.durbin_watson(df_1["AAPL_LOG"]))


# In[16]:


#Series Plot
df_1["AAPL_LOG"].plot(figsize=(16,9))
plt.show()


# In[17]:


#Autocorrelation Plot
fig = plt.figure(figsize=(16,9))
ax1 = fig.add_subplot(211)
fig = sm.graphics.tsa.plot_acf(df_1["AAPL_LOG"].values.squeeze(), lags=35, ax=ax1)
ax2 = fig.add_subplot(212)
fig = sm.graphics.tsa.plot_pacf(df_1["AAPL_LOG"], lags=35, ax=ax2)


# In[18]:


#Getting the 'AAPL_LOG' column values as array with dropping NaN values
array_1 = (df_1["AAPL_LOG"].dropna().as_matrix())


# In[19]:


#Creating a column 'AAPL_LOG_DIFF' with data as difference of 'AAPL_LOG' column current row and previous row
df_1["AAPL_LOG_DIFF"] = df_1["AAPL_LOG"] - df_1["AAPL_LOG"].shift(periods=-1)


# In[20]:


#Creating ARMA Model
model_1 = sm.tsa.ARMA(array_1,(2,0)).fit()
print(model_1.params) #Prints model parameter


# In[21]:


#Printing Model's AIC, BIC and HQIC values
print(model_1.aic, model_1.bic, model_1.hqic)


# In[22]:


#Finding the best values for ARIMA model parameter
#aic=999999
aic=99
a,b,c = 0,0,0

for p in range(3):
    for q in range(1,3):
        for r in range(3):
            try:
                model= ARIMA(array_1,(p,q,r)).fit()
                if(aic > model_1.aic):
                    aic = model_1.aic
                    a,b,c = p,q,r
            except:
                pass
                
print(a,b,c)


# In[23]:


#Creating and fitting ARIMA model
model_1_arima = ARIMA(array_1,(0, 1, 0)).fit()


# In[24]:


#Positive serial correlation is associated with Durbin-Watson values below 2 and negative serial correlation with Durbin-Watson values above 2. 
#Prints Durbin-Watson statistic of given data.
print("Durbin-Watson statistic:",sm.stats.durbin_watson(model_1_arima.resid))


# In[25]:


#Predicting the values using ARIMA Model
pred_1 = model_1_arima.predict()
pred_1


# ### Root Mean Squared Error

# In[26]:


#Printing RMSE value for the model
print(np.sqrt(mean_squared_error(pred_1,df_1["AAPL_LOG_DIFF"][:-1])))


# ## 2. NASDAQ.ADP

# In[27]:


df_2 = df.copy() #Makes a copy of df dataframe.


# In[28]:


#Creating a column 'ADP_LOG' with the log values of 'NASDAQ.ADP' column data
df_2["ADP_LOG"] = df_2["NASDAQ.ADP"].apply(lambda x:np.log(x)) 


# In[29]:


df_2.head() #Returns the first 5 rows of df_2 dataframe


# In[30]:


type(df_2["DATE"][0]) #Type of the 'DATE' column


# In[31]:


#Creating a new column 'DATE_NEW' with formatted timestamp 
df_2["DATE_NEW"] = df_2["DATE"].apply(lambda x:dt.datetime.fromtimestamp(x).strftime("%Y-%m-%d %H:%M:%S"))


# In[32]:


df_2.head() #Returns the first 5 rows of df_2 dataframe


# In[33]:


#Positive serial correlation is associated with Durbin-Watson values below 2 and negative serial correlation with Durbin-Watson values above 2. 
#Prints Durbin-Watson statistic of given data.
print("Durbin-Watson statistic:",sm.stats.durbin_watson(df_2["ADP_LOG"]))


# In[34]:


#Series Plot
df_2["ADP_LOG"].plot(figsize=(16,9))
plt.show()


# In[35]:


#Autocorrelation Plot
fig = plt.figure(figsize=(16,9))
ax1 = fig.add_subplot(211)
fig = sm.graphics.tsa.plot_acf(df_2["ADP_LOG"].values.squeeze(), lags=35, ax=ax1)
ax2 = fig.add_subplot(212)
fig = sm.graphics.tsa.plot_pacf(df_2["ADP_LOG"], lags=35, ax=ax2)


# In[36]:


#Getting the 'AAPL_LOG' column values as array with dropping NaN values
array_2 = (df_2["ADP_LOG"].dropna().as_matrix())


# In[37]:


#Creating a column 'ADP_LOG_DIFF' with data as difference of 'ADP_LOG' column current row and previous row
df_2["ADP_LOG_DIFF"] = df_2["ADP_LOG"] - df_2["ADP_LOG"].shift(periods=-1)


# In[38]:


#Creating ARMA Model
model_2 = sm.tsa.ARMA(array_2,(2,0)).fit()
print(model_2.params) #Prints model parameter


# In[39]:


#Printing Model's AIC, BIC and HQIC values
print(model_2.aic, model_2.bic, model_2.hqic)


# In[40]:


#Finding the best values for ARIMA model parameter
#aic=999999
aic=99
a,b,c = 0,0,0

for p in range(3):
    for q in range(1,3):
        for r in range(3):
            try:
                model= ARIMA(array_2,(p,q,r)).fit()
                if(aic > model_2.aic):
                    aic = model_2.aic
                    a,b,c = p,q,r
            except:
                pass
                
print(a,b,c)


# In[41]:


#Creating and fitting ARIMA model
model_2_arima = ARIMA(array_2,(0, 1, 0)).fit()


# In[42]:


#Positive serial correlation is associated with Durbin-Watson values below 2 and negative serial correlation with Durbin-Watson values above 2. 
#Prints Durbin-Watson statistic of given data.
print("Durbin-Watson statistic:",sm.stats.durbin_watson(model_2_arima.resid))


# In[43]:


#Predicting the values using ARIMA Model
pred_2 = model_2_arima.predict()
pred_2


# ### Root Mean Squared Error

# In[44]:


#Printing RMSE value for the model
print(np.sqrt(mean_squared_error(pred_2,df_2["ADP_LOG_DIFF"][:-1])))


# ## 3. NASDAQ.CBOE

# In[45]:


df_3 = df.copy() #Makes a copy of df dataframe.


# In[46]:


#Creating a column 'CBOE_LOG' with the log values of 'NASDAQ.CBOE' column data
df_3["CBOE_LOG"] = df_3["NASDAQ.CBOE"].apply(lambda x:np.log(x)) 


# In[47]:


df_3.head() #Returns the first 5 rows of df_3 dataframe


# In[48]:


type(df_3["DATE"][0]) #Type of the 'DATE' column


# In[49]:


#Creating a new column 'DATE_NEW' with formatted timestamp 
df_3["DATE_NEW"] = df_3["DATE"].apply(lambda x:dt.datetime.fromtimestamp(x).strftime("%Y-%m-%d %H:%M:%S"))


# In[50]:


df_3.head() #Returns the first 5 rows of df_3 dataframe


# In[51]:


#Positive serial correlation is associated with Durbin-Watson values below 2 and negative serial correlation with Durbin-Watson values above 2. 
#Prints Durbin-Watson statistic of given data.
print("Durbin-Watson statistic:",sm.stats.durbin_watson(df_3["CBOE_LOG"]))


# In[52]:


#Series Plot
df_3["CBOE_LOG"].plot(figsize=(16,9))
plt.show()


# In[53]:


#Autocorrelation Plot
fig = plt.figure(figsize=(16,9))
ax1 = fig.add_subplot(211)
fig = sm.graphics.tsa.plot_acf(df_3["CBOE_LOG"].values.squeeze(), lags=35, ax=ax1)
ax2 = fig.add_subplot(212)
fig = sm.graphics.tsa.plot_pacf(df_3["CBOE_LOG"], lags=35, ax=ax2)


# In[54]:


#Getting the 'CBOE_LOG' column values as array with dropping NaN values
array_3 = (df_3["CBOE_LOG"].dropna().as_matrix())


# In[55]:


#Creating a column 'CBOE_LOG_DIFF' with data as difference of 'CBOE_LOG' column current row and previous row 
df_3["CBOE_LOG_DIFF"] = df_3["CBOE_LOG"] - df_3["CBOE_LOG"].shift(periods=-1)


# In[56]:


#Creating ARMA Model
model_3 = sm.tsa.ARMA(array_3,(2,0)).fit()
print(model_3.params) #Prints model parameter


# In[57]:


#Printing Model's AIC, BIC and HQIC values
print(model_3.aic, model_3.bic, model_3.hqic)


# In[58]:


#Finding the best values for ARIMA model parameter
#aic=999999
aic=99
a,b,c = 0,0,0

for p in range(3):
    for q in range(1,3):
        for r in range(3):
            try:
                model= ARIMA(array_3,(p,q,r)).fit()
                if(aic > model_3.aic):
                    aic = model_3.aic
                    a,b,c = p,q,r
            except:
                pass
                
print(a,b,c)


# In[59]:


#Creating and fitting ARIMA model
model_3_arima = ARIMA(array_3,(0, 1, 0)).fit()


# In[60]:


#Positive serial correlation is associated with Durbin-Watson values below 2 and negative serial correlation with Durbin-Watson values above 2. 
#Prints Durbin-Watson statistic of given data.
print("Durbin-Watson statistic:",sm.stats.durbin_watson(model_3_arima.resid))


# In[61]:


#Predicting the values using ARIMA Model
pred_3 = model_3_arima.predict()
pred_3


# ### Root Mean Squared Error

# In[62]:


#Printing RMSE value for the model
print(np.sqrt(mean_squared_error(pred_3,df_3["CBOE_LOG_DIFF"][:-1])))


# ## 4. NASDAQ.CSCO

# In[63]:


df_4 = df.copy() #Makes a copy of df dataframe.


# In[64]:


#Creating a column 'CSCO_LOG' with the log values of 'NASDAQ.CSCO' column data
df_4["CSCO_LOG"] = df_4["NASDAQ.CSCO"].apply(lambda x:np.log(x)) 


# In[65]:


df_4.head() #Returns the first 5 rows of df_4 dataframe


# In[66]:


type(df_4["DATE"][0]) #Type of the 'DATE' column


# In[67]:


#Creating a new column 'DATE_NEW' with formatted timestamp 
df_4["DATE_NEW"] = df_4["DATE"].apply(lambda x:dt.datetime.fromtimestamp(x).strftime("%Y-%m-%d %H:%M:%S"))


# In[68]:


df_4.head() #Returns the first 5 rows of df_4 dataframe


# In[69]:


#Positive serial correlation is associated with Durbin-Watson values below 2 and negative serial correlation with Durbin-Watson values above 2. 
#Prints Durbin-Watson statistic of given data.
print("Durbin-Watson statistic:",sm.stats.durbin_watson(df_4["CSCO_LOG"]))


# In[70]:


#Series Plot
df_4["CSCO_LOG"].plot(figsize=(16,9))
plt.show()


# In[71]:


#Autocorrelation Plot
fig = plt.figure(figsize=(16,9))
ax1 = fig.add_subplot(211)
fig = sm.graphics.tsa.plot_acf(df_4["CSCO_LOG"].values.squeeze(), lags=35, ax=ax1)
ax2 = fig.add_subplot(212)
fig = sm.graphics.tsa.plot_pacf(df_4["CSCO_LOG"], lags=35, ax=ax2)


# In[72]:


#Getting the 'CSCO_LOG' column values as array with dropping NaN values
array_4 = (df_4["CSCO_LOG"].dropna().as_matrix())


# In[73]:


#Creating a column 'AAPL_LOG_DIFF' with data as difference of 'AAPL_LOG' column current row and previous row
df_4["CSCO_LOG_DIFF"] = df_4["CSCO_LOG"] - df_4["CSCO_LOG"].shift(periods=-1)


# In[74]:


#Creating ARMA Model
model_4 = sm.tsa.ARMA(array_4,(2,0)).fit()
print(model_4.params) #Prints model parameter


# In[75]:


#Printing Model's AIC, BIC and HQIC values
print(model_4.aic, model_4.bic, model_4.hqic)


# In[76]:


#Finding the best values for ARIMA model parameter
#aic=9999
aic=99
a,b,c = 0,0,0

for p in range(3):
    for q in range(1,3):
        for r in range(3):
            try:
                model= ARIMA(array_4,(p,q,r)).fit()
                if(aic > model_4.aic):
                    aic = model_4.aic
                    a,b,c = p,q,r
            except:
                pass
                
print(a,b,c)


# In[77]:


#Creating and fitting ARIMA model
model_4_arima = ARIMA(array_4,(0, 1, 0)).fit()


# In[78]:


#Positive serial correlation is associated with Durbin-Watson values below 2 and negative serial correlation with Durbin-Watson values above 2. 
#Prints Durbin-Watson statistic of given data.
print("Durbin-Watson statistic:",sm.stats.durbin_watson(model_4_arima.resid))


# In[79]:


#Predicting the values using ARIMA Model
pred_4 = model_4_arima.predict()
pred_4


# ### Root Mean Squared Error

# In[80]:


#Printing RMSE value for the model
print(np.sqrt(mean_squared_error(pred_4,df_4["CSCO_LOG_DIFF"][:-1])))


# ## 5. NASDAQ.EBAY

# In[81]:


df_5 = df.copy() #Makes a copy of df dataframe.


# In[82]:


#Creating a column 'EBAY_LOG' with the log values of 'NASDAQ.EBAY' column data
df_5["EBAY_LOG"] = df_5["NASDAQ.EBAY"].apply(lambda x:np.log(x)) 


# In[83]:


df_5.head() #Returns the first 5 rows of df_5 dataframe


# In[84]:


type(df_5["DATE"][0]) #Type of the 'DATE' column


# In[85]:


#Creating a new column 'DATE_NEW' with formatted timestamp 
df_5["DATE_NEW"] = df_5["DATE"].apply(lambda x:dt.datetime.fromtimestamp(x).strftime("%Y-%m-%d %H:%M:%S"))


# In[86]:


df_5.head() #Returns the first 5 rows of df_5 dataframe


# In[87]:


#Positive serial correlation is associated with Durbin-Watson values below 2 and negative serial correlation with Durbin-Watson values above 2. 
#Prints Durbin-Watson statistic of given data.
print("Durbin-Watson statistic:",sm.stats.durbin_watson(df_5["EBAY_LOG"]))


# In[88]:


#Series Plot
df_5["EBAY_LOG"].plot(figsize=(16,9))
plt.show()


# In[89]:


#Autocorrelation Plot
fig = plt.figure(figsize=(16,9))
ax1 = fig.add_subplot(211)
fig = sm.graphics.tsa.plot_acf(df_5["EBAY_LOG"].values.squeeze(), lags=35, ax=ax1)
ax2 = fig.add_subplot(212)
fig = sm.graphics.tsa.plot_pacf(df_5["EBAY_LOG"], lags=35, ax=ax2)


# In[90]:


#Getting the 'EBAY_LOG' column values as array with dropping NaN values
array_5 = (df_5["EBAY_LOG"].dropna().as_matrix())


# In[91]:


#Creating a column 'EBAY_LOG_DIFF' with data as difference of 'EBAY_LOG' column row and previous row
df_5["EBAY_LOG_DIFF"] = df_5["EBAY_LOG"] - df_5["EBAY_LOG"].shift(periods=-1)


# In[92]:


#Creating ARMA Model
model_5 = sm.tsa.ARMA(array_5,(2,0)).fit()
print(model_5.params) #Prints model parameter


# In[93]:


#Printing Model's AIC, BIC and HQIC values
print(model_5.aic, model_5.bic, model_5.hqic)


# In[94]:


#Finding the best values for ARIMA model parameter
#aic=999999
aic=99
a,b,c = 0,0,0

for p in range(3):
    for q in range(1,3):
        for r in range(3):
            try:
                model= ARIMA(array_5,(p,q,r)).fit()
                if(aic > model_5.aic):
                    aic = model_5.aic
                    a,b,c = p,q,r
            except:
                pass
                
print(a,b,c)


# In[95]:


#Creating and fitting ARIMA model
model_5_arima = ARIMA(array_5,(0, 1, 0)).fit()


# In[96]:


#Positive serial correlation is associated with Durbin-Watson values below 2 and negative serial correlation with Durbin-Watson values above 2. 
#Prints Durbin-Watson statistic of given data.
print("Durbin-Watson statistic:",sm.stats.durbin_watson(model_5_arima.resid))


# In[97]:


#Predicting the values using ARIMA Model
pred_5 = model_5_arima.predict()
pred_5


# ### Root Mean Squared Error

# In[98]:


#Printing RMSE value for the model
print(np.sqrt(mean_squared_error(pred_5,df_5["EBAY_LOG_DIFF"][:-1])))

