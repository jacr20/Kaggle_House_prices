import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn import preprocessing
from sklearn.preprocessing import OneHotEncoder
from scipy import stats
import math
from collections import Counter
import scipy.stats as ss


'''Functions'''
def missing_replace(col_list, value_list):
    '''Function to find values in a dataframe that are missing for one or more feature, replaces them with a specific value. Id column sould NOT be passed.'''
    if 'Id' in col_list:
        raise Exception('You are attempting to pass "Id" as a list argument, this should not be done with missing_replace.')
    if len(col_list) == 1:
        for i in train_data.loc[train_data[col_list[0]].isnull()]['Id']:
            for t in range(len(col_list)):
                 train_data.at[(i-1), col_list[t]] = value_list[t]
    elif len(col_list)==2:
        for i in train_data.loc[(train_data[col_list[0]]).isnull() & (train_data[col_list[1]].isnull())]['Id']:
            for t in range(len(col_list)):
                train_data.at[(i-1), col_list[t]] = value_list[t]
    elif len(col_list)==3:
        for i in train_data.loc[(train_data[col_list[0]]).isnull() & (train_data[col_list[1]].isnull()) & (train_data[col_list[2]].isnull())]['Id']:
            for t in range(len(col_list)):
                train_data.at[(i-1), col_list[t]] = value_list[t]
    elif len(col_list)==4:
        for i in train_data.loc[(train_data[col_list[0]]).isnull() & (train_data[col_list[1]].isnull()) & (train_data[col_list[2]].isnull())& (train_data[col_list[3]].isnull())]['Id']:
            for t in range(len(col_list)):
                train_data.at[(i-1), col_list[t]] = value_list[t]
    else:
        for i in train_data.loc[(train_data[col_list[0]]).isnull() & (train_data[col_list[1]].isnull()) & (train_data[col_list[2]].isnull()) & (train_data[col_list[3]].isnull()) & (train_data[col_list[4]].isnull())]['Id']:
            for t in range(len(col_list)):
                train_data.at[(i-1), col_list[t]] = value_list[t]

    return train_data

def cat_mode_replace(list_var, spec_val):
    '''Function to replace missing values in a column with the modal value of the feature in another categorical feature.
    The first value of the feature list is the name of the column with the missing values. The second value of the list is the featue to find the mode with.'''
    dfCMR = train_data[list_var].groupby(list_var[1])
    dfCMR_ = dfCMR[list_var[0]].apply(lambda x:x.mode())
    if type(spec_val) == list:
        for t in range(len(spec_val)):
            train_data.at[spec_val[t], list_var[0]] = dfCMR_[(train_data[list_var[1]][spec_val[t]]),0]
    else:
        for i in train_data.loc[(train_data[list_var[0]].isnull())]['Id']:
            train_data.at[(i-1), list_var[0]] = dfCMR_[(train_data[list_var[1]][(i-1)]),0]
            
    return train_data

def cat_median_replace(list_var, spec_val):
    '''Function to replace missing values in a column with the median value of the feature in another category.
    The first value of the feature list is the name of the column with the missing values. The second value of the list is the featue to find the mode with.'''
    dfCMeR = train_data[list_var].groupby(list_var[1])
    dfCMeR_ = dfCMeR[list_var[0]].apply(lambda x:x.median())
    if type(spec_val) == list:
        for t in range(len(spec_val)):
            train_data.at[spec_val[t], list_var[0]] = dfCMeR_[(train_data[list_var[1]][spec_val[t]])]
    else:
        for i in train_data.loc[(train_data[list_var[0]].isnull())]['Id']:
            train_data.at[(i-1), list_var[0]] = dfCMeR_[(train_data[list_var[1]][(i-1)])]
    
    return train_data



train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
test['SalePrice'] =  'NA'
train_data = pd.concat([train, test], ignore_index=True)

'''Maps- for ordinal label encoding'''
Qualities = {'None': 0, 'Po':1, 'Fa': 2, 'TA': 3, 'Gd':4, 'Ex':5}
LotShape_map = {'IR3':0, 'IR2':1, 'IR1':2, 'Reg':3}
Finish_map  = {'None':0, 'Unf':1, 'RFn':2, 'Fin':3}
Exposure = {'None':0, 'No':1, 'Mn':2, 'Av':3, 'Gd':4}
FinType = {'None':0, 'Unf':1, 'LwQ':2, 'Rec':3, 'BLQ':4, 'ALQ':5, 'GLQ':6}
Masonry = {'None':0, 'BrkCmn':1, 'BrkFace':2, 'Stone':3}
Function = {'Sal':0, 'Sev':1, 'Maj2':2, 'Maj1':3, 'Mod':4, 'Min2':5, 'Min1':6, 'Typ':7}
central  = {'N':0, 'Y':1}
slope = {'Sev':0, 'Mod':1, 'Gtl':2}
streets = {'Grvl':0, 'Pave':1}
pave = {'N':0, 'P':1, 'Y':2}

'''Garage'''
train_data['GarageQual'] = train_data['GarageQual'].map(Qualities)
train_data['GarageCond'] = train_data['GarageCond'].map(Qualities)
train_data['GarageFinish'] = train_data['GarageFinish'].map(Finish_map)

Gar_list = ['GarageType', 'GarageFinish', 'GarageQual', 'GarageCond']
Gar_value = ['None', 0, 0, 0]
train_data['GarageYrBlt']=train_data['GarageYrBlt'].fillna(train_data['YearBuilt'].mode()[0])
train_data.at[2126, 'GarageCond'] = train_data['GarageCond'].mode()
train_data.at[2126, 'GarageQual'] = train_data['GarageQual'].mode()
train_data.at[2576, 'GarageArea'] = 0
train_data.at[2576, 'GarageCars'] = 0
train_data.at[2576, 'GarageType'] = np.NaN
missing_replace(Gar_list, Gar_value)
train_data.at[2126, 'GarageFinish'] = train_data['GarageFinish'].mode()

'''Lot Footage'''
listLF = ['LotFrontage', 'Neighborhood']
specVal = None
cat_median_replace(listLF, specVal)


'''Pool'''
train_data.at[2420, 'PoolQC'] = 'Fa'
train_data.at[2503, 'PoolQC'] = 'TA'
train_data.at[2599, 'PoolQC'] = 'Fa'
train_data['PoolQC'] = train_data['PoolQC'].fillna('None')

'''Bsmt'''
Bsmt1 = ['BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2']
Bsmt2 = ['None', 'None', 'None', 'None', 'None']
missing_replace(Bsmt1, Bsmt2)
Bsmt_list1 = ['BsmtQual', 'BsmtFinType1']
value1 = [2217,2218]
cat_mode_replace(Bsmt_list1, value1)
Bsmt_list2 = ['BsmtCond', 'BsmtFinType1']
value2 = [2040, 2185, 2524]
cat_mode_replace(Bsmt_list2, value2)
Bsmt_list3 = ['BsmtExposure', 'Neighborhood']
value3 = [948, 1487, 2348]
cat_mode_replace(Bsmt_list3, value3)
Bsmt_list4 = ['BsmtFinType2', 'Neighborhood']
value4 = [332]
cat_mode_replace(Bsmt_list4, value4)

for i in train_data.loc[(train_data['BsmtFullBath'].isnull()) | (train_data['BsmtHalfBath'].isnull()) | (train_data['BsmtFinSF1'].isnull()) | (train_data['BsmtFinSF2'].isnull()) | (train_data['BsmtUnfSF'].isnull()) | (train_data['TotalBsmtSF'].isnull())]['Id']:
    if (i-1) == 2120:
        train_data.at[(i-1), 'BsmtFullBath'] = 0
        train_data.at[(i-1), 'BsmtHalfBath'] = 0
        train_data.at[(i-1), 'BsmtFinSF1'] = 0
        train_data.at[(i-1), 'BsmtFinSF2'] = 0
        train_data.at[(i-1), 'BsmtUnfSF'] = 0
        train_data.at[(i-1), 'TotalBsmtSF'] = 0
    else:
        train_data.at[(i-1), 'BsmtFullBath'] = 0
        train_data.at[(i-1), 'BsmtHalfBath'] = 0

'''Masonary Variables'''
MasList1 = ['MasVnrType','MasVnrArea']
Mas_val2 = ['None', 0]
missing_replace(MasList1, Mas_val2)
df_MV = train_data[['Id','MasVnrType','MasVnrArea']]
for i in train_data.loc[(df_MV['MasVnrType'].isnull()) & (df_MV['MasVnrArea'].notnull())]['Id']:
    train_data.at[(i-1), 'MasVnrType'] = train_data['MasVnrType'].value_counts().index[1]
df_MVA = df_MV.groupby(['MasVnrType'])
df_MVA_med = df_MVA['MasVnrArea'].apply(lambda x: x.median())
for i in train_data.loc[(df_MV['MasVnrType']!= 'None') & (df_MV['MasVnrArea']==0)]['Id']:
    Name = train_data['MasVnrType'][(i-1)]   
    train_data.at[(i-1), 'MasVnrArea'] = df_MVA_med[Name]

    

df_MV = train_data[['Id','MasVnrType','MasVnrArea']]
for i in df_MV.loc[(df_MV['MasVnrType'] == 'None') & (df_MV['MasVnrArea']!=0)]['Id']:
    train_data.at[(i-1), 'MasVnrArea'] = 0
    
MS_list1 = ['MSZoning', 'Neighborhood','Id']
MS_value = None
cat_mode_replace(MS_list1, MS_value)
'''Kitchen'''
Kitch_list = ['KitchenQual','OverallQual','Id']
value = None
cat_mode_replace(Kitch_list, value)

'''Utilities'''
train_data = train_data.drop(['Utilities'],axis = 1)
'''Functional'''
train_data['Functional']=train_data['Functional'].fillna(train_data['Functional'].mode()[0])

'''Exterior'''
Ext_list1 = ['Exterior1st', 'Neighborhood', 'Id']
Ext_val = None
cat_mode_replace(Ext_list1, Ext_val)
Ext_list2 = ['Exterior2nd', 'Exterior1st', 'Id']
Ext_val2 = None
cat_mode_replace(Ext_list2, Ext_val2)

'''Electrical'''
train_data['Electrical']=train_data['Electrical'].fillna(train_data['Electrical'].mode()[0])

'''SaleType'''
train_data['SaleType']=train_data['SaleType'].fillna(train_data['SaleType'].mode()[0])

'''Alley'''
for col in ('Alley','FireplaceQu','Fence','MiscFeature'):
    train_data[col]=train_data[col].fillna('None')

#print("Skewness: %f" % train_data['SalePrice'].skew())
#print("Kurtosis: %f" % train_data['SalePrice'].kurt())

#print(train_data.loc[(train_data['PoolQC'] =='None') & (train_data['PoolArea'] > 0)])

'''Label Encoding'''

train_data['PoolQC'] = train_data['PoolQC'].map(Qualities)
train_data['FireplaceQu'] = train_data['FireplaceQu'].map(Qualities)
train_data['LotShape'] = train_data['LotShape'].map(LotShape_map)
train_data['BsmtQual'] = train_data['BsmtQual'].map(Qualities)
train_data['BsmtCond'] = train_data['BsmtCond'].map(Qualities)
train_data['BsmtExposure'] = train_data['BsmtExposure'].map(Exposure)
train_data['BsmtFinType1'] = train_data['BsmtFinType1'].map(FinType)
train_data['BsmtFinType2'] = train_data['BsmtFinType2'].map(FinType)
train_data['MasVnrType'] = train_data['MasVnrType'].map(Masonry)
train_data['KitchenQual'] = train_data['KitchenQual'].map(Qualities)
train_data['Functional'] = train_data['Functional'].map(Function)
train_data['ExterQual'] = train_data['ExterQual'].map(Qualities)
train_data['ExterCond'] = train_data['ExterCond'].map(Qualities)
train_data['HeatingQC'] = train_data['HeatingQC'].map(Qualities)
train_data['CentralAir'] = train_data['CentralAir'].map(central)
train_data['LandSlope'] = train_data['LandSlope'].map(slope)
train_data['Street'] = train_data['Street'].map(streets)
train_data['PavedDrive'] = train_data['PavedDrive'].map(pave)

'''One-hot encoding'''
TRAIN = train_data[0:1460]
TEST = train_data[1460:]

X_train_cat = TRAIN.select_dtypes(exclude =['int','float']).drop('SalePrice',axis = 1)
Y_train_cat = TEST.select_dtypes(exclude =['int','float']).drop('SalePrice',axis = 1)

enc = OneHotEncoder(sparse = False, handle_unknown='ignore')
Ohe = pd.DataFrame(enc.fit_transform(X_train_cat),columns = enc.get_feature_names())
TRAIN = TRAIN.drop(X_train_cat.columns, axis = 1)
XY_train =  pd.concat([TRAIN, Ohe], axis  =0)

Ohe1 = pd.DataFrame(enc.transform(Y_train_cat),columns = enc.get_feature_names())
TEST = TEST.drop(Y_train_cat.columns, axis = 1)
XY_test  =  pd.concat([TEST, Ohe1], axis  = 0)

print(XY_train.select_dtypes(exclude =['int','float']).columns)
print(XY_test.select_dtypes(exclude =['int','float']).columns)
