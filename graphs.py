import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import math
from collections import Counter
from sklearn import preprocessing
import modules_graphs.py as mg


train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
test['SalesPrice'] =  'NA'
train_data = pd.concat([train, test], ignore_index=True)



'''Theil'''
theilu = pd.DataFrame(index = ['SalPrice'],columns=train_data.columns)
columns = train_data.columns
for j in range(0,len(columns)):
    u = mg.theil_u(train_data['SalePrice'].tolist(),train_data[columns[j]].tolist())
    theilu.loc[:,columns[j]] = u
theilu.fillna(value=np.nan,inplace=True)
plt.figure(figsize=(20,1))
sns.heatmap(theilu.T.nlargest(10,'SalePrice'),annot=True,fmt='.2f')
plt.yticks(rotation=45)
plt.show()

'''correlation matrix'''

corrmat = train_data.corr()
k = 20 #number of variables for heatmap
cols = corrmat.nlargest(k, 'SalePrice')['SalePrice'].index
cm = np.corrcoef(train_data[cols].values.T)
sns.set(font_scale=1.25)
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)
plt.show()





'''Missing data'''
total2 = train_data.select_dtypes(include=['int','float']).isnull().sum().sort_values(ascending=False)
total = train_data.isnull().sum().sort_values(ascending=False)
percent = (train_data.isnull().sum()/train_data.isnull().count()).sort_values(ascending=False)
percent2 = (train_data.select_dtypes(include=['int','float']).isnull().sum()/train_data.select_dtypes(include=['int','float']).isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total missing', 'Percent missing'])
missing_cont = pd.concat([total2, percent2], axis=1, keys=['Total missing', 'Percent missing'])
f, ax = plt.subplots(figsize=(15, 6))
plt.xticks(rotation='90')
sns.barplot(x=missing_cont.index, y=missing_cont['Percent missing'])
plt.xlabel('Features', fontsize=15)
plt.ylabel('Percent of missing values', fontsize=15)
plt.title('Percent missing continuous data by feature', fontsize=15)
plt.show()




