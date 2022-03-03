#!/usr/bin/env python
# coding: utf-8

# # Dataset level analysis:
# 
# Data Analysis is one of the most important steps in solving any Machine Learning problem.
# As the very first step, let’s import the required libraries to solve this problem.
# 

# In[1]:


import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import StandardScaler
import numpy as np # linear algebra
import os # accessing directory structure
from scipy import stats
from scipy.stats import zscore
#from IPython import get_ipython


# In[2]:


#%matplotlib inline
#pd.options.display.float_format = '{:.2f}'.format


# ### Loading the data:
# 
# Data set is in the format of a csv file. Load the data from “telecom_churn_data.csv” to a pandas dataframe.
# 

# In[3]:


# read the data

churn_data = pd.read_csv("telecom_churn_data.csv")


# In[4]:


# shape of data
churn_data.shape


# There are 99999 data points(rows) and 226 features(columns) in dataset.
#                                               

# In[5]:


# Display all Columns of data
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.max_colwidth', None)


# In[6]:


#print first five rows of data
churn_data.head(5)


# In[7]:


# columns of dataset

churn_data.columns


# In[8]:


print(churn_data.info())


# #### observaton:
# There are 214  numeric  and  12 non numeric columns.

# In[9]:


# observe the data type of  each column

churn_data.info(verbose = True)


# #### observation:
# The 12 non numeric columns have datetime values.  
# columns with date time values represented as object,they can be converted into date_time format.

# In[10]:


# print  columns with datetime represented as object 

date = churn_data.select_dtypes(include=['object'])
print(" columns with date time format represented as  object\n",date.columns)

#converting these dateime cols to datetime format
for col in date.columns:
    churn_data[col] = pd.to_datetime(churn_data[col])


# In[11]:


# datatypes after converting to datetime columns
churn_data.info(verbose = True)


# In[12]:


# some statistical information about features.
churn_data.describe().T.head()


# #### preprocessing data:

# In[13]:


# check for unique values in data
unique_col = [col for col in churn_data.columns if churn_data[col].nunique() == 1]

print(len(unique_col))

print( "\n columns with unique value in data set:","\n", unique_col)


# observation:
# 
# There are 16 unique value columns. These columns have no variance in data. so we drop those unique value columns.

# In[14]:


# Drop the unique value columns
churn_data.drop(unique_col, axis=1, inplace = True)


# In[15]:


# shape of data after dropimg  unique columns
churn_data.shape


# ### Handling missing values

# In[16]:


#checking for missing values in data
churn_data.isna().any().any()


#  There are lot of missing values in data. These missing values can be handled by using some imputation techniques.

# In[17]:


# checking for percentage of missing in each column
missing = churn_data.isnull().sum() * 100 / churn_data.shape[0]


# no of missing values in each columna
number_missing = churn_data.isnull().sum()


# In[18]:


# print missing values and % of missing in a data frame
missing_columns = pd.DataFrame({'column': churn_data.columns,
                               'No_of_missing_val':number_missing,
                               'Missing_percentage':missing}).set_index('column').reset_index()
missing_columns


# In[19]:


#print missing percentage in each column.
percent_missing = churn_data.isna().sum()/(len(churn_data))*100
percent_missing.sort_values(ascending = False)


# #### observation

# Lot of columns missing over 74% of data. impute those values by observing the data in each column.
# some columns filled with  0 and some filled with 1

# In[20]:


# checking for value counts of night pack user

print(churn_data['night_pck_user_6'].value_counts(),"\n" ,
      churn_data['night_pck_user_7'].value_counts())


#  over 97% of night_pack values are zeros .Hence we fill these nan values with zeros
# 
# 
# 
# 

# In[21]:


# fill nan values with zeros

night_pck_cols = [col for col in churn_data.columns if 'night_pck_user_' in col]
print(night_pck_cols)

churn_data[night_pck_cols] = churn_data[night_pck_cols].apply(lambda x: x.fillna(0))

churn_data[night_pck_cols].head(5)


# In[22]:


# checking for value counts fb_user


print(churn_data['fb_user_6'].value_counts(),"\n" ,
      churn_data['fb_user_7'].value_counts())


#  over 90% of the values of fb_user columns  are 1. Hence we fill nan values with 1's.

# In[23]:


# Fill nan values with 1
fb_user_cols = [col for col in churn_data.columns if 'fb_user_' in col]
print(fb_user_cols)

churn_data[fb_user_cols] = churn_data[fb_user_cols].apply(lambda x: x.fillna(1))

churn_data[fb_user_cols].head(5)


# In[24]:


churn_data.isnull().sum() * 100 / churn_data.shape[0]
churn_data.shape


# In[25]:


print(churn_data['total_rech_data_6'].min(),"\n" ,
      churn_data['total_rech_data_7'].min())


# In[26]:


total_rech_cols = [col for col in churn_data.columns if 'total_rech_data_' in col]
print(total_rech_cols)

churn_data[total_rech_cols] = churn_data[total_rech_cols].apply(lambda x: x.fillna(0))

churn_data[total_rech_cols].head(5)


# In[27]:


print(churn_data['max_rech_data_8'].min(),"\n" ,
      churn_data['max_rech_data_7'].min())


# In[28]:


max_rech_cols = [col for col in churn_data.columns if 'max_rech_data_' in col]
print(max_rech_cols)

churn_data[max_rech_cols] = churn_data[max_rech_cols].apply(lambda x: x.fillna(0))

churn_data[max_rech_cols].head(5)


#  Filling the maximum recharge data and total recharge data with 0 means customer didn't make any recharge.

# In[29]:


# list of some recharge columns
rech_cols = []
for col in churn_data.columns:
    if 'count_rech_' in col or 'av_rech_amt_' in col:
        rech_cols.append(col)
rech_cols


# In[30]:


# impute these rech_cols  with zero
churn_data[rech_cols] = churn_data[rech_cols].apply(lambda x: x.fillna(0))


# In[31]:


churn_data[rech_cols].head()


# In[32]:


percent_missing = churn_data.isna().sum()/(len(churn_data))*100
percent_missing.sort_values(ascending = False)


#  By observing the date columns 74% of 'date_of_last_rech_data_', 'date_of_last_rech_' column  values missing.
# we can fill those with  first date of every month.

# In[33]:


# Date of last recharge column fill with first date of month
churn_data['date_of_last_rech_data_6'].fillna('6/1/2014',inplace=True)
churn_data['date_of_last_rech_data_7'].fillna('7/1/2014',inplace=True)
churn_data['date_of_last_rech_data_8'].fillna('8/1/2014',inplace=True)
churn_data['date_of_last_rech_data_9'].fillna('9/1/2014',inplace=True)


# In[34]:


# Date of last recharge column fill with first date of month
churn_data['date_of_last_rech_6'].fillna('6/1/2014',inplace=True)
churn_data['date_of_last_rech_7'].fillna('7/1/2014',inplace=True)
churn_data['date_of_last_rech_8'].fillna('8/1/2014',inplace=True)
churn_data['date_of_last_rech_9'].fillna('9/1/2014',inplace=True)


# In[35]:


# Checking the related columns values
churn_data[['arpu_3g_6','arpu_2g_6','av_rech_amt_data_6']].head(10)


# In[36]:



arpu = churn_data.iloc[:,1:4]


# In[37]:


# Checking the correlation between the above mentioned columns in tabular for months 6,7,8 and 9

print("Correlation  for month 6\n\n")
print( churn_data[['arpu_3g_6','arpu_2g_6','av_rech_amt_data_6']].corr())
print("\nCorrelation  for month 7")
print( churn_data[['arpu_3g_7','arpu_2g_7','av_rech_amt_data_7']].corr())
print("\nCorrelation  for month 8")
print( churn_data[['arpu_3g_8','arpu_2g_8','av_rech_amt_data_8']].corr())
print("\nCorrelation for month 9")
print(churn_data[['arpu_3g_9','arpu_2g_9','av_rech_amt_data_9']].corr())


#  From the above correlation table between attributes arpu_2g_* and arpu_3g_* for each month from 6 to 9 respectively
#  is highly correlated to the attribute av_rech_amt_data_* for each month from 6 to 9 respectively.
# 
# Considering the high correlation between them, it is safer to drop the attributes arpu_2g_* and arpu_3g_*.
# 
# 

# In[38]:


arpu_cols = []
for col in churn_data.columns:
    if 'arpu_3g_' in col or 'arpu_2g_' in col:
        arpu_cols.append(col)
arpu_cols


# In[39]:


churn_data.drop(arpu_cols, axis = 1, inplace = True)


# In[40]:


# shape of data after droping arpu_ columns
churn_data.shape


# In[41]:


# print missing values after droping some columns
percent_missing = churn_data.isna().sum()/(len(churn_data))*100
percent_missing.sort_values(ascending = False)


# In[42]:


#churn_data.describe().T


# ### Filtering high value customers

# High value customers can be filtered by using the revenue generated by them in previous months.

# In[43]:


# Calculating the total recharge amount done for data alone in months 6,7,8 and 9
churn_data['total_rech_amt_data_6'] = churn_data['av_rech_amt_data_6'] * churn_data['total_rech_data_6']
churn_data['total_rech_amt_data_7'] = churn_data['av_rech_amt_data_7'] * churn_data['total_rech_data_7']
churn_data['total_rech_amt_data_8'] = churn_data['av_rech_amt_data_8'] * churn_data['total_rech_data_8']
churn_data['total_rech_amt_data_9'] = churn_data['av_rech_amt_data_9'] * churn_data['total_rech_data_9']

drop_col = ["total_rech_data_6", "total_rech_data_7", "total_rech_data_8", "total_rech_data_9", 
               "av_rech_amt_data_6", 'av_rech_amt_data_7', 'av_rech_amt_data_8', 'av_rech_amt_data_9']
churn_data.drop(drop_col, axis=1, inplace=True)

print(churn_data.shape)


# In[44]:


# Calculating the overall recharge amount for the months 6,7,8 and 9
churn_data['overall_rech_amt_6'] = churn_data['total_rech_amt_data_6'] + churn_data['total_rech_amt_6']
churn_data['overall_rech_amt_7'] = churn_data['total_rech_amt_data_7'] + churn_data['total_rech_amt_7']


# Calculating the average recharge done by customer in months 6 and 7
churn_data['avg_rech_amt_6_7'] = (churn_data['overall_rech_amt_6'] + churn_data['overall_rech_amt_7'])/2


# In[45]:


percent_missing = churn_data.isna().sum()/(len(churn_data))*100
percent_missing.sort_values(ascending = False)


# In[46]:


churn_data.shape


# # Define churn variable

# In[47]:


# Selecting the columns to define churn variable as  TARGET Variable

churn_col=['total_ic_mou_9','total_og_mou_9','vol_2g_mb_9','vol_3g_mb_9']

print(churn_col)
churn_data[churn_col].info()


# In[48]:



# Add churn column to data
churn_data['churn']=np.nan

# Imputing the churn values based on the condition
churn_data['churn'] = np.where(churn_data[churn_col].sum(axis=1) == 0, 1, 0)

churn_data['churn'].head()


# In[49]:


# lets find out churn/non churn percentage
print((churn_data['churn'].value_counts()/len(churn_data))*100)
((churn_data['churn'].value_counts()/len(churn_data))*100).plot(kind="pie")
plt.title("churn_distribution")
plt.xlabel("churn_value")
plt.ylabel("churn_count")
plt.show()


# ### observation:
# By observing the above graph over 92% of customers are not churned. only 8% are in churned state.

# In[50]:


churn_data.shape


# In[51]:


churn_cols = [col for col in churn_data.columns if '_9' in col]
print("The columns from churn phase are:\n",churn_cols)


# In[52]:


# Dropping the selected churn phase columns
churn_data.drop(churn_cols, axis=1, inplace=True)

# The curent dimension of the dataset after dropping the churn related columns
churn_data.shape


# In[53]:


percent_missing = churn_data.isna().sum()/(len(churn_data))*100
percent_missing.sort_values(ascending = False)


# In[54]:


#  reduce some columns by  Merging local og calls
churn_data['loc_og_mou_t_6']=churn_data['loc_og_t2t_mou_6']+churn_data['loc_og_t2m_mou_6']+churn_data['loc_og_t2f_mou_6']+churn_data['loc_og_t2c_mou_6']
churn_data['loc_og_mou_t_7']=churn_data['loc_og_t2t_mou_7']+churn_data['loc_og_t2m_mou_7']+churn_data['loc_og_t2f_mou_7']+churn_data['loc_og_t2c_mou_7']
churn_data['loc_og_mou_t_8']=churn_data['loc_og_t2t_mou_8']+churn_data['loc_og_t2m_mou_8']+churn_data['loc_og_t2f_mou_8']+churn_data['loc_og_t2c_mou_8']

churn_data.drop(columns=['loc_og_t2t_mou_6','loc_og_t2t_mou_7','loc_og_t2t_mou_8','loc_og_t2m_mou_6','loc_og_t2m_mou_7','loc_og_t2m_mou_8','loc_og_t2f_mou_6','loc_og_t2f_mou_7','loc_og_t2f_mou_8','loc_og_t2c_mou_6','loc_og_t2c_mou_7','loc_og_t2c_mou_8'],inplace=True)


# In[55]:


churn_data.shape


# In[56]:


#percent_missing = churn_data.isna().sum()/(len(churn_data))*100
#percent_missing.sort_values(ascending = False)


# In[57]:



# Merging the std_og calls
churn_data['std_og_mou_t_6']=churn_data['std_og_t2t_mou_6']+churn_data['std_og_t2m_mou_6']+churn_data['std_og_t2f_mou_6']
churn_data['std_og_mou_t_7']=churn_data['std_og_t2t_mou_7']+churn_data['std_og_t2m_mou_7']+churn_data['std_og_t2f_mou_7']
churn_data['std_og_mou_t_8']=churn_data['std_og_t2t_mou_8']+churn_data['std_og_t2m_mou_8']+churn_data['std_og_t2f_mou_8']

churn_data.drop(columns=['std_og_t2t_mou_6','std_og_t2t_mou_7','std_og_t2t_mou_8','std_og_t2m_mou_6','std_og_t2m_mou_7','std_og_t2m_mou_8','std_og_t2f_mou_6','std_og_t2f_mou_7','std_og_t2f_mou_8'],inplace=True)

# Merging the local ic calls
churn_data['loc_ic_mou_t_6']=churn_data['loc_ic_t2t_mou_6']+churn_data['loc_ic_t2m_mou_6']+churn_data['loc_ic_t2f_mou_6']
churn_data['loc_ic_mou_t_7']=churn_data['loc_ic_t2t_mou_7']+churn_data['loc_ic_t2m_mou_7']+churn_data['loc_ic_t2f_mou_7']
churn_data['loc_ic_mou_t_8']=churn_data['loc_ic_t2t_mou_8']+churn_data['loc_ic_t2m_mou_8']+churn_data['loc_ic_t2f_mou_8']

churn_data.drop(columns=['loc_ic_t2t_mou_6','loc_ic_t2t_mou_7','loc_ic_t2t_mou_8','loc_ic_t2m_mou_6','loc_ic_t2m_mou_7','loc_ic_t2m_mou_8','loc_ic_t2f_mou_6','loc_ic_t2f_mou_7','loc_ic_t2f_mou_8'],inplace=True)

# Merging the std ic calls
churn_data['std_ic_mou_t_6']=churn_data['std_ic_t2t_mou_6']+churn_data['std_ic_t2m_mou_6']+churn_data['std_ic_t2f_mou_6']
churn_data['std_ic_mou_t_7']=churn_data['std_ic_t2t_mou_7']+churn_data['std_ic_t2m_mou_7']+churn_data['std_ic_t2f_mou_7']
churn_data['std_ic_mou_t_8']=churn_data['std_ic_t2t_mou_8']+churn_data['std_ic_t2m_mou_8']+churn_data['std_ic_t2f_mou_8']

churn_data.drop(columns=['std_ic_t2t_mou_6','std_ic_t2t_mou_7','std_ic_t2t_mou_8','std_ic_t2m_mou_6','std_ic_t2m_mou_7','std_ic_t2m_mou_8','std_ic_t2f_mou_6','std_ic_t2f_mou_7','std_ic_t2f_mou_8'],inplace=True)


# In[58]:


churn_data.shape


# In[59]:


import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(25, 20))
sns.heatmap(churn_data.corr(), annot=True)

plt.show()



#   By observing the above heatmap lot of features are correlated.

# In[60]:


churn_data.shape


# In[61]:


# Correlation of "Churn" with other variables:
plt.figure(figsize=(40,10))
churn_data.corr()['churn'].sort_values(ascending = False).plot(kind='bar')


# ### observation:
# By observing the above plot the features total_ic_mou,total_rech_num,max_rech_amt,total_og_mou,
#    last_day_rech_amt of month 8 are negatively correlated with churn variable.
#   std_og_mou,onnet_mou,roam_og_mou,offnet_mou features of moth 6 are positively correlated with churn.

# ### univariate analysis

# In[62]:


# Distribution plot of total_og_mou_8
sns.FacetGrid(churn_data, hue = 'churn', height = 5)    .map(sns.distplot,'total_og_mou_8')    .add_legend()
plt.show()


# #### observation:
# churned customers make less number of outgoing minutes of usage in 8 th month.

# In[63]:


#distribution plot
sns.FacetGrid(churn_data, hue = 'churn', height = 5)    .map(sns.distplot,'std_og_mou_8')    .add_legend()
plt.show()


# In[64]:


# distribution plot of onnet_mou_6
sns.FacetGrid(churn_data, hue = 'churn', height = 5)    .map(sns.distplot,'onnet_mou_6')    .add_legend()
plt.show()


# #### observation:
# overlap between churn and non churn customers of netusage in month 6

# In[65]:


# distribution of 3g net usage

sns.FacetGrid(churn_data, hue = 'churn', height = 5)    .map(sns.distplot,'vol_2g_mb_8',bins = 10)    .add_legend()
plt.show()


# #### observation:
#    churned customers use less 2g mobile intenet usage in month 8 almost close top zero.

# In[66]:


# plot of vol_3g_mb_8
sns.FacetGrid(churn_data, hue = 'churn', height = 5)    .map(sns.distplot,'vol_3g_mb_8')    .add_legend()
plt.show()


# #### observation:
#     churned customers use less 3g mobile intenet usage in month 8 almost close top zero.

# In[67]:


# plot of arpu_8
sns.FacetGrid(churn_data, hue = 'churn', height = 5)    .map(sns.distplot,'arpu_8')    .add_legend()
plt.show()


# Most of the churned customers have produced a low revenue close to 0 in the eighth month. 
# This feature can help slightly in distinguish but still a lot  overlapping between both.
# 
# 

# In[68]:


# distribution of aon
sns.FacetGrid(churn_data, hue = 'churn', height = 5)    .map(sns.distplot,'aon')    .add_legend()
plt.show()


# lot of overlap in churned and non churned by using aon.
# 

# In[69]:


# plot of total_rech_num_8
sns.FacetGrid(churn_data, hue = 'churn', height = 5)    .map(sns.distplot,'total_rech_num_8')    .add_legend()
plt.show()


# 
# churned customers make less number of recharges in month 8.

# In[70]:


# plot of max_rech_amt_8
plt.figure(figsize=(8,5))
sns.boxplot( x = 'churn', y = 'max_rech_amt_8',data = churn_data)
plt.show()


# #### observation:
#     churned customers make less maximum recharge amount  in month 8.

# In[71]:


sns.FacetGrid(churn_data, hue = 'churn', height = 5)    .map(sns.distplot,'max_rech_amt_8')    .add_legend()
plt.show()


# #### observation:
#     churned customers make lowest recharge amount in month 8 almost equal to zero.

# In[72]:


arpu = churn_data.iloc[:,1:4]
#arpu
arpu['churn'] = churn_data.churn


# In[73]:


# bivariate distributions in a dataset, use pair plots 


import seaborn as sns
plt.figure(figsize = (4,2))
sns.pairplot(arpu,hue = 'churn', height=2,diag_kind='hist')
plt.show()


# #### observation:
#      In the above pair plots, there is lot of overlaping.The churned customers have made a good revenue in one 
#     month and not in other month.The diagonal histogram shows that most of the churned customers have values close to 0.
# 
# 

# In[74]:


total_og=churn_data.iloc[:,31:34]
total_og['churn'] = churn_data.churn


# In[75]:


# Bivariate analysis of total_og and churn
sns.pairplot(total_og, hue="churn", height=3,diag_kind='hist')


# #### observation:
#     lot of overlaping in months 6 and 7.  
#     By observing diagonal plots,total_og_mou of  churned customers for month 8 is lmost equal to zero.

# In[76]:


# plot of aon and churn
sns.boxplot(y = 'aon', x = 'churn', data = churn_data)

plt.show()


# #### observation:
# From the above plot , its clear that  tenured customers(who stay along network from long time) do not churn much.
# 
# 

# In[77]:


churn_data.head()


# In[78]:


#arpu = [col for col in churn_data.columns if 'arpu_' in col]

arpu = churn_data.iloc[:,1:4]
arpu['churn'] = churn_data.churn


# In[79]:


# plot of arpu_6 and churn
plt.figure(figsize=(8,5))

sns.boxplot(x = 'churn', y = 'arpu_6', data = arpu)
plt.show()


# ##### observation:
#     lot of overlaping. Mean values looks like same for churned and non churned customers.

# In[80]:


# plot for total_og_mou_8
plt.figure(figsize=(8,5))


sns.boxplot( x = 'churn', y = 'total_og_mou_8',data = churn_data)
plt.show()


#  The total outgoing minutes of usage for churned customers almost equal to zero.

# #### observations:
#     
#      For features like incoming,outgoing,recharge most of the churned customers have low values. Especially in the eighth month
#      
#      Recharge amounts distinguish churned customers better than the other features
#      
#      The pair plots show that the most of the churned customers have close to 0 value. 
# 
#      Multiple features representing the same task
# 
# 
# 
# 
# 

# ### Featuring engineering

# ######  creating some new features based on existing ones for better analysis of data.

# In[81]:


# separating date columns from dataframe
date_cols = []
for col in churn_data.columns:
    if 'date_of_last_rech_data_' in col or 'date_of_last_rech_' in col:
        date_cols.append(col)
date_cols


# In[82]:


# convert date filelds into date time format

churn_data['date_of_last_rech_6']=pd.to_datetime(churn_data['date_of_last_rech_6'])
churn_data['date_of_last_rech_7']=pd.to_datetime(churn_data['date_of_last_rech_7'])
churn_data['date_of_last_rech_8']=pd.to_datetime(churn_data['date_of_last_rech_8'])

churn_data['date_of_last_rech_data_6']=pd.to_datetime(churn_data['date_of_last_rech_data_6'])

churn_data['date_of_last_rech_data_7']=pd.to_datetime(churn_data['date_of_last_rech_data_7'])
churn_data['date_of_last_rech_data_8']=pd.to_datetime(churn_data['date_of_last_rech_data_8'])







# In[83]:


#extracting date features
churn_data['last_rech_date_6']=churn_data['date_of_last_rech_6'].dt.day
churn_data['last_rech_date_7']=churn_data['date_of_last_rech_7'].dt.day
churn_data['last_rech_date_8']=churn_data['date_of_last_rech_8'].dt.day

churn_data['last_data_rech_date_6']=churn_data['date_of_last_rech_data_6'].dt.day

churn_data['last_data_rech_date_7']=churn_data['date_of_last_rech_data_7'].dt.day

churn_data['last_data_rech_date_8']=churn_data['date_of_last_rech_data_8'].dt.day



#  we can't use datetime as it is. so we extract day from date filelds.
# 

# In[84]:


# drop date_cols, we already extract date features
churn_data.drop(date_cols, axis = 1, inplace = True)


# In[85]:


churn_data.head()
churn_data.shape


# In[86]:


# creating new feature using difference between arpu_8 and avg of 6and 7 th months

churn_data['arpu_diff'] = churn_data['arpu_8']- (churn_data[['arpu_6','arpu_7']].mean(axis=1))


# arpu_diff column will calculate the difference between arpu of 8 th month and avg arpu of 6 and 7 months.
# This arpu_diff shows the customer behaviour.

# In[87]:


churn_data.shape


#  The final shape of our data is (99999,128). This wiil be used for model building.

# ## Model building

# In[88]:


#create data set for model building

df_model = churn_data[:].copy()
#df_model.head()


# In[89]:


# Drop the column mobile number
df_model.drop('mobile_number', axis=1, inplace=True)

# fill the remaining fields
df_model.fillna(0, inplace = True)
df_model.head()


# In[90]:


#df_model.to_csv(index = False)


# In[91]:


df_model.shape


# In[92]:


# Drop the churn column for prediction

X = df_model.drop(['churn'], axis=1)
y = df_model['churn']

df_model.drop('churn', axis=1, inplace=True)


# In[93]:


df_model.head()


# In[94]:


df_model.shape


# In[95]:


import sklearn
print(sklearn.__version__)


# In[96]:


X.info()


# In[ ]:





# #### splitting the data

# In[97]:


# splitting the data into test and train

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(
        X, y, test_size = 0.3, random_state = 0)


# In[98]:


# shape of data after splitting
print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)


# In[99]:


print(sum(y_train==0))
print(sum(y_train==1))


# In[100]:


from sklearn.dummy import DummyClassifier

# Initialize Estimator
dummy_clf = DummyClassifier(strategy='stratified')
dummy_clf.fit(x_train,y_train)

# Check for Model Accuracy
dummy_clf.score(x_train,y_train)


# In[101]:


print(dummy_clf.score(x_train,y_train))
print(dummy_clf.score(x_test,y_test))


# #### Balance the data

# In[102]:


y_train_imb = (y_train != 0).sum()/(y_train == 0).sum()
y_test_imb = (y_test != 0).sum()/(y_test == 0).sum()
print("Imbalance in Train Data : ", y_train_imb)
print("Imbalance in Test Data : ", y_test_imb)


# In[103]:


x_train.shape


# In[104]:


y_train.shape


# In[105]:


#pip install imbalanced-learn


# In[106]:


# Balancing DataSet
#from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import SMOTE


sm = SMOTE(random_state=42)
x_tr,y_tr = sm.fit_resample(x_train,y_train)


# In[107]:


print(x_tr.shape)
print(y_tr.shape)
print(x_test.shape)
print("count label 0:",sum(y_tr==0))
print("count label 1:",sum(y_tr==1))


# In[108]:


from sklearn.dummy import DummyClassifier

# Initialize Estimator
dummy_clf = DummyClassifier(strategy='stratified')
dummy_clf.fit(x_tr,y_tr)

# Check for Model Accuracy
dummy_clf.score(x_tr,y_tr)


# In[109]:


print(dummy_clf.score(x_tr,y_tr))
print(dummy_clf.score(x_test,y_test))


# In[ ]:





# ### XGB model

# In[112]:


import xgboost as xgb 
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV

#x_tr = np.array(x_tr)



param_grid = {
    'n_estimators' : [ 200],
    'learning_rate' : [  0.1, 0.2, 0.3],
    'max_depth' : [2],
    'sub_sample' : [0.2,0.3,0.4]
}
xgb = xgb.XGBClassifier()


grid_search = GridSearchCV(estimator=xgb, param_grid=param_grid, cv=5, scoring='roc_auc',verbose = 1,n_jobs = -1)

grid_search.fit(x_tr, y_tr)


#search.best_params_


# In[113]:


print(grid_search.best_params_)


# In[114]:


import time
start_time = time.time()
model = XGBClassifier( max_depth=2,
                     learning_rate = 0.3,
                    n_estimators = 200,
                   sub_sample = 0.2)
model.fit(x_tr, y_tr)

end_time = time.time()
print("total time:",end_time - start_time)


# In[115]:


# feature_importance
importance = model.feature_importances_

# create dataframe
feature_importance = pd.DataFrame({'variables': X.columns, 'importance_percentage': importance*100})
feature_importance = feature_importance[['variables', 'importance_percentage']]

# sort features
feature_importance = feature_importance.sort_values('importance_percentage', ascending=False).reset_index(drop=True)
print("Sum of importance=", feature_importance.importance_percentage.sum())
feature_importance


# In[116]:


# extract top 'n' features
top_n = 30
top_features = feature_importance.variables[0:top_n]
top_features.values


# In[117]:


# plot feature correlation
import seaborn as sns
plt.rcParams["figure.figsize"] =(8,8)
mycmap = sns.diverging_palette(199, 359, s=99, center="light", as_cmap=True)
sns.heatmap(data=x_tr[top_features].corr(), center=0.0, cmap=mycmap)


# In[118]:


top_features = ['loc_ic_mou_t_8','total_ic_mou_8', 'roam_og_mou_8','arpu_diff', 'total_rech_amt_8', 'loc_ic_mou_8','last_rech_date_8', 'roam_ic_mou_8','count_rech_2g_8' ]
x_out = x_tr[top_features]
x_test = x_test[top_features]


# In[119]:


y_tr.values


# In[ ]:


len(y_tr.values)


# In[120]:


x_out = np.array(x_out)
x_out.shape


# In[121]:


model.fit(x_out, y_tr)


# In[122]:


#model with top most features
import time
start_time = time.time()
model = XGBClassifier( max_depth=2,
                     learning_rate = 0.3,
                    n_estimators = 200,
                   sub_sample = 1)
model.fit(x_out, y_tr)

end_time = time.time()
print("total time:",end_time - start_time)


# In[ ]:





# In[123]:


from sklearn.model_selection import GridSearchCV


# In[124]:


print(" xgboost AUC score on training data:",model.score(x_out,y_tr))


# In[125]:


x_tr[top_features].columns


# In[126]:


x_test = x_test[top_features]
x_test.head()


# In[127]:


x_test = np.array(x_test)


# In[128]:


y_pred = model.predict(x_test)


# In[129]:


from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score

print("AUC: ", roc_auc_score(y_test, y_pred))

#print ("Accuracy : ", accuracy_score(y_test, y_pred))


# In[130]:


from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

cm = confusion_matrix(y_test, y_pred, labels=model.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm,  
                              display_labels=model.classes_)
disp.plot()

plt.show()


# In[131]:


import sklearn.metrics as metrics

y_pred_proba = model.predict_proba(x_test)[::,1]
fpr, tpr, _ = metrics.roc_curve(y_test,  y_pred_proba)
auc = metrics.roc_auc_score(y_test, y_pred_proba)
plt.plot(fpr,tpr,label="data 1, auc="+str(auc))

plt.title('Receiver Operating Characteristic Curve', fontsize=16)
plt.ylabel('TPR', fontsize=16)
plt.xlabel('FPR', fontsize=16)

plt.legend(loc=4)
plt.show()


# In[ ]:





#  By observingthe above  models  xgboost with hyperparameters give best AUC.

# #### saving model using pickle

# In[132]:


import joblib
joblib.dump(model,'model_final.pkl')


# In[133]:


import joblib
# Load the model from the file 
loaded_model = joblib.load('model_final.pkl') 
# Use the loaded model to make predictions
response_model = loaded_model.predict(x_test)
response_model.shape


# In[134]:


from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


# ### Evaluate the loaded model

# In[135]:


import time
start_time = time.time()

# predict churn on test data
y_pred_load = loaded_model.predict(x_test)
end_time = time.time()
print("total time:",end_time - start_time)
print(y_pred_load)
# create onfusion matrix
cm_matrix = confusion_matrix(y_test, y_pred_load)
disp = ConfusionMatrixDisplay(confusion_matrix=cm_matrix,  
                              display_labels=model.classes_)
disp.plot()

plt.show()

#print(cm)
# check area under curve
y_pred_prob_load = loaded_model.predict_proba(x_test)[:, 1]
print("ROC:    \t", round(roc_auc_score(y_test, y_pred_prob_load),2))


# ### Evaluation of trained model object

# In[136]:


#predict churn
y_predict = model.predict(x_test)
print('Accuracy: {}'.format(accuracy_score(y_test, y_predict)))


print(confusion_matrix(y_test,y_predict))
disp = ConfusionMatrixDisplay(confusion_matrix=cm,  
                              display_labels=model.classes_)
disp.plot()
plt.show()
# check area under curve
y_predict = model.predict(x_test)
print("ROC:    \t", round(roc_auc_score(y_test, y_pred_prob_load),2))


# In[ ]:




