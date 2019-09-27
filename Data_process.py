import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn import preprocessing
from scipy import stats
import math
from collections import Counter
import scipy.stats as ss

'''Process the training data - What does the data look like'''
'''training data'''

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
test['SalesPrice'] =  'NA'
train_data = pd.concat([train, test], ignore_index=True)

train_data['GarageYrBlt']=train_data['GarageYrBlt'].fillna(train_data['YearBuilt'].mode()[0])

train_data.at[2126, 'GarageFinish'] = train_data['GarageFinish'].mode()
train_data.at[2126, 'GarageCond'] = train_data['GarageCond'].mode()
train_data.at[2126, 'GarageQual'] = train_data['GarageQual'].mode()

train_data.at[2576, 'GarageArea'] = 0
train_data.at[2576, 'GarageCars'] = 0
train_data.at[2576, 'GarageType'] = np.NaN



'''
for i in range(len(train_data)):
    if (train_data['LotFrontage'].isnull()[i] == True).all():
        df = train_data[['LotFrontage','Neighborhood']].groupby(['Neighborhood']).median()
        train_data['LotFrontage'][i] = df['LotFrontage'][(train_data['Neighborhood'][i])]
'''
print('--------------------------------------------------------------------------')
#print(train_data.loc[(train_data['PoolQC'] =='None') & (train_data['PoolArea'] > 0)])
print('--------------------------------------------------------------------------')

for i in train_data.loc[(train_data['BsmtQual'].isnull()) & (train_data['BsmtCond'].isnull()) & (train_data['BsmtExposure'].isnull()) & (train_data['BsmtFinType1'].isnull()) & (train_data['BsmtFinType2'].isnull())]['Id']:
    train_data.at[(i-1), 'BsmtQual'] = 0
    train_data.at[(i-1), 'BsmtCond'] = 0
    train_data.at[(i-1), 'BsmtExposure'] = 0
    train_data.at[(i-1), 'BsmtFinType1'] = 0
    train_data.at[(i-1), 'BsmtFinType2'] = 0


df = train_data[['Id','Neighborhood','BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinSF1','BsmtFinType2','BsmtFinSF2','BsmtUnfSF','TotalBsmtSF','BsmtFullBath','BsmtHalfBath']]
#print(pd.concat([df.loc[df['BsmtQual'].isnull()], df.loc[df['BsmtCond'].isnull()], df.loc[df['BsmtExposure'].isnull()], df.loc[df['BsmtFinType1'].isnull()], df.loc[df['BsmtFinType2'].isnull()]],ignore_index=True).head(25))

dfQ = train_data[['BsmtQual','BsmtFinType1']].groupby(['BsmtFinType1'])
dfQual = dfQ['BsmtQual'].apply(lambda x: x.mode())
train_data.at[2217, 'BsmtQual'] = dfQual[(train_data['BsmtFinType1'][2217]),0]
train_data.at[2218, 'BsmtQual'] = dfQual[(train_data['BsmtFinType1'][2218]),0]

dfQ = train_data[['BsmtCond','BsmtFinType1']].groupby(['BsmtFinType1'])
dfQual = dfQ['BsmtCond'].apply(lambda x: x.mode())
train_data.at[2040, 'BsmtCond'] = dfQual[(train_data['BsmtFinType1'][2040]),0]
train_data.at[2185, 'BsmtCond'] = dfQual[(train_data['BsmtFinType1'][2185]),0]
train_data.at[2524, 'BsmtCond'] = dfQual[(train_data['BsmtFinType1'][2524]),0]

dfQ = train_data[['BsmtExposure','Neighborhood']].groupby(['Neighborhood'])
dfQual = dfQ['BsmtExposure'].apply(lambda x: x.mode())
print(dfQual.head())
train_data.at[948, 'BsmtExposure'] = dfQual[(train_data['Neighborhood'][948]),0]
train_data.at[1487, 'BsmtExposure'] = dfQual[(train_data['Neighborhood'][1487]),0]
train_data.at[2348, 'BsmtExposure'] = dfQual[(train_data['Neighborhood'][2348]),0]

dfQ = train_data[['BsmtFinType2','Neighborhood']].groupby(['Neighborhood'])
dfQual = dfQ['BsmtFinType2'].apply(lambda x: x.mode())
train_data.at[332, 'BsmtFinType2'] = dfQual[(train_data['Neighborhood'][332]),0]

df = train_data[['Id','Neighborhood','BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinSF1','BsmtFinType2','BsmtFinSF2','BsmtUnfSF','TotalBsmtSF','BsmtFullBath','BsmtHalfBath']]
print(pd.concat([df.loc[df['BsmtQual'].isnull()], df.loc[df['BsmtCond'].isnull()], df.loc[df['BsmtExposure'].isnull()], df.loc[df['BsmtFinType1'].isnull()], df.loc[df['BsmtFinType2'].isnull()]],ignore_index=True).head(25))


'''
print(df.isnull().sum().head(12))
'''
print('--------------------------------------------------------------------------')







#df.loc[df['BsmtFinSF1'].isnull()], df.loc[df['BsmtFinSF2'].isnull()], df.loc[df['BsmtUnfSF'].isnull()], df.loc[df['TotalBsmtSF'].isnull()], df.loc[df['BsmtFullBath'].isnull(),df.loc[df['BsmtHalfBath'].isnull()


#print(train_data.loc[train_data['MasVnrType'].isnull()])



#count = train_data['MasVnrType'].value_counts()
#sumc = count.sum()
#print(train_data['MasVnrType'].isnull().sum())

print('--------------------------------------------------------------------------')
#print(train_data.iloc[332,:])
#print(count)

print('--------------------------------------------------------------------------')

#print(train_data['MasVnrType'].isnull().sum() )
#print(sumc +train_data['MasVnrType'].isnull().sum() )

print('--------------------------------------------------------------------------')


print('--------------------------------------------------------------------------')

'''
count = train_data['BldgType'].value_counts()
sumc = count.sum()
print(count)
print(train_data['['MasVnrType']'].isnull().sum() )
print(sumc +train_data['BldgType'].isnull().sum() )
'''
'''
#for these collums, the value isn't missing, it should be 'None'.
for col in ('Alley','Utilities','MasVnrType','BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1',
            'BsmtFinType2','Electrical','FireplaceQu','GarageType','GarageFinish','GarageQual','GarageCond',
           'PoolQC','Fence','MiscFeature'):
    train_data[col]=train_data[col].fillna('None')

for col in ('MasVnrArea','BsmtFinSF1','BsmtFinSF2','BsmtUnfSF','TotalBsmtSF','BsmtFullBath','BsmtHalfBath','GarageYrBlt','GarageCars','GarageArea'):
    train_data[col]=train_data[col].fillna(0)
    #test[col]=test[col].fillna(0)

#for these collums, the value is missing - it must be filled. As categorical data, the mode of the rest of the collum was used. 
for col in ('Exterior1st','Exterior2nd','SaleType','Functional'):
    train_data[col]=train_data[col].fillna(train_data[col].mode()[0])

#MsZone_by_ExtQual = train_data.groupby(['ExterQual'], as_index=False).median()['MSZoning']
# Filling the missing values in Age with the medians of Sex and Pclass groups
train_data['MSZoning'] = train_data.groupby(['ExterQual'])['MSZoning'].apply(lambda x: x.fillna(x.value_counts().index[0]))
#MsZone_by_ExtQual = train_data.groupby(['ExterQual','HeatingQC','BsmtQual'], as_index=False).value_counts().index[0]['KitchenQual']
#print(MsZone_by_ExtQual)
# Filling the missing values in Age with the medians of Sex and Pclass groups
train_data['KitchenQual'] = train_data.groupby(['ExterQual','HeatingQC','BsmtQual'])['KitchenQual'].apply(lambda x: x.fillna(x.value_counts().index[0]))

train_data['LotFrontage']=train_data['LotFrontage'].fillna(train_data['LotFrontage'].median())

#total = train_data.isnull().sum().sort_values(ascending=False)
#percent = (train_data.isnull().sum()/train_data.isnull().count()).sort_values(ascending=False)
#missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
#print(missing_data.head(20))
'''
print('--------------------------------------------------------------------------')
'''
train_data1 = train_data.select_dtypes(include=[object])

for feature in train_data1.columns:
    uniq = np.unique(train_data1[feature])
    print('{}: {} distinct values -  {}'.format(feature,len(uniq),uniq))
'''
print('--------------------------------------------------------------------------')

def conditional_entropy(x,y):
    # entropy of x given y
    y_counter = Counter(y)
    xy_counter = Counter(list(zip(x,y)))
    total_occurrences = sum(y_counter.values())
    entropy = 0
    for xy in xy_counter.keys():
        p_xy = xy_counter[xy] / total_occurrences
        p_y = y_counter[xy[1]] / total_occurrences
        entropy += p_xy * math.log(p_y/p_xy)
    return entropy

def theil_u(x,y):
    s_xy = conditional_entropy(x,y)
    x_counter = Counter(x)
    total_occurrences = sum(x_counter.values())
    p_x = list(map(lambda n: n/total_occurrences, x_counter.values()))
    s_x = ss.entropy(p_x)
    if s_x == 0:
        return 1
    else:
        return (s_x - s_xy) / s_x
'''
theilu = pd.DataFrame(index = ['SalPrice'],columns=train_data.columns)
columns = train_data.columns
for j in range(0,len(columns)):
    u = theil_u(train_data['SalePrice'].tolist(),train_data[columns[j]].tolist())
    theilu.loc[:,columns[j]] = u
theilu.fillna(value=np.nan,inplace=True)
plt.figure(figsize=(20,1))
sns.heatmap(theilu.T.nlargest(10,'SalePrice'),annot=True,fmt='.2f')
plt.yticks(rotation=45)
plt.show()
'''
print('--------------------------------------------------------------------------')

#saleprice correlation matrix

'''
#correlation matrix
corrmat = train_data.corr()
k = 20 #number of variables for heatmap
cols = corrmat.nlargest(k, 'SalePrice')['SalePrice'].index
cm = np.corrcoef(train_data[cols].values.T)
sns.set(font_scale=1.25)
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)
#plt.yticks(rotation=45)
#plt.xticks(rotation=45)
plt.show()


'''
#count = train_data['KitchenQual'].value_counts()
#sumc = count.sum()
#print(train_data['KitchenQual'].isnull().sum())
#print(train_data.loc[train_data['MasVnrType'].isnull()])

print('--------------------------------------------------------------------------')

#print(sumc)
#print(count)

print('--------------------------------------------------------------------------')

#print(train_data.loc[train_data['MasVnrType'].isnull()])
#print(train_data.isnull().sum().sum())

'''
df_train_cor = train_data.drop(['LotArea','MasVnrArea','BsmtFinSF1','BsmtFinSF2','BsmtUnfSF','TotalBsmtSF','1stFlrSF','2ndFlrSF',
'LowQualFinSF','GrLivArea','BsmtFullBath','BsmtHalfBath','FullBath','HalfBath','TotRmsAbvGrd','GarageCars','GarageArea','WoodDeckSF','OpenPorchSF',
'EnclosedPorch','3SsnPorch','ScreenPorch','PoolArea','MiscVal', 'MoSold','YrSold'], axis=1)

corrs = df_train_cor.corr(method='spearman')[['KitchenQual']]#[['MSZoning', 'Exterior1st','Exterior2nd','KitchenQual','SaleType','Functional','LotFrontage']]
print(corrs.info())
plt.figure(figsize=[60,30])
g=sns.heatmap(corrs, annot=True)
plt.yticks(rotation=45)
plt.show()

'''
#sns.distplot(train_data['SalePrice'], label='Sale price', color='#2ecc71')
#sns.distplot(np.log(train_data['SalePrice']), label='Sale price', color='#2ecc71')
#plt.show()
'''Kurtosis - indicates outliers in the distrbution. '''
print("Skewness: %f" % train_data['SalePrice'].skew())
print("Kurtosis: %f" % train_data['SalePrice'].kurt())

#print(train_data.loc[(train_data['PoolQC'] =='None') & (train_data['PoolArea'] > 0)])


'''
enc = preprocessing.OneHotEncoder()
enc.fit(XO_2)

# 3. Transform
onehotlabels = enc.transform(XO_2).toarray()
XOH = pd.DataFrame(onehotlabels)
print(type(onehotlabels))
print(XOH.head())

\
XI = train_data.select_dtypes(include=[int,float])
#print(XI.head())

#print(train_data.head())
'''


