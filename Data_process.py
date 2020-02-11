import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import xgboost as xgb
from xgboost.sklearn import XGBClassifier
from matplotlib.pylab import rcParams
import numpy as np
import random
import seaborn as sns
import sklearn
from sklearn import preprocessing
import sklearn.ensemble
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso, Ridge #, ridge_regression
from sklearn.model_selection import GridSearchCV
from scipy import stats
import math
from collections import Counter
import scipy.stats as ss

np.random.seed(0)
random.seed(0)

'''Functions'''

def modelfit(alg, train, target, useTrainCV=True, cv_folds=5, early_stopping_rounds=50):
    
    if useTrainCV:
        xgb_param = alg.get_xgb_params()
        xgtrain = xgb.DMatrix(train.values, label=target.values)
        cvresult = xgb.cv(xgb_param, xgtrain, num_boost_round=alg.get_params()['n_estimators'], nfold=cv_folds,
            metrics='rmse', early_stopping_rounds=early_stopping_rounds, seed = 0, verbose_eval = 5)
        alg.set_params(n_estimators=cvresult.shape[0])
        print(cvresult.shape[0])
        print(cvresult)
    #Fit the algorithm on the data
    alg.fit(train, target, eval_metric='rmse')
        
    #Predict training set:
    dtrain_predictions = alg.predict(train)
    
    #Print model report:
    print("\nModel Report")
    print("MSE : %.4g" % sklearn.metrics.mean_squared_error(target.values, dtrain_predictions))
    
    feat_imp = pd.Series(alg.get_booster().get_fscore()).sort_values(ascending=False)
    feat_imp.plot(kind='bar', title='Feature Importances')
    plt.ylabel('Feature Importance Score')
    plt.show()

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

pd.set_option('mode.chained_assignment', None)
pd.options.display.max_columns = None

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
test['SalePrice'] =  'NA'
train_data = pd.concat([train, test], ignore_index=True)
test_labels = test['Id']

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

'''Outliers'''
train_data.at[(2592), 'GarageYrBlt'] = 2007
train_data.drop([523, 1298], axis = 0, inplace = True)

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

'''Feature engineering - New variables'''
train_data['TotalBath'] = train_data['FullBath'] + train_data['BsmtFullBath'] + (0.5*train_data['HalfBath']) +(0.5* train_data['BsmtHalfBath'])

train_data['Remodeled'] = ''
for i in train_data['Id']:
    if train_data.at[(i-1),'YearBuilt'] == train_data.at[(i-1), 'YearRemodAdd']:
        train_data.at[(i-1) ,'Remodeled'] = 1
    else:
        train_data.at[(i-1) ,'Remodeled'] = 0

train_data['Age'] = train_data['YrSold'] - train_data['YearRemodAdd']

train_data['IsNew'] = ''
for i in train_data['Id']:
    if (train_data.at[(i-1),'YrSold'] - train_data.at[(i-1), 'YearBuilt'])< 5:
        train_data.at[(i-1) ,'IsNew'] = 1
    else:
        train_data.at[(i-1) ,'IsNew'] = 0

train_data['TotalSF'] = train_data['GrLivArea'] + train_data['TotalBsmtSF']
train_data['TotalPorchSF'] = train_data['3SsnPorch'] + train_data['OpenPorchSF'] + train_data['EnclosedPorch'] + train_data['ScreenPorch'] 
train_data['GrLivArea'] = train_data['1stFlrSF'] + train_data['2ndFlrSF'] + train_data['LowQualFinSF']
train_data['TotalBsmtSF'] = train_data['BsmtFinSF1'] + train_data['BsmtFinSF2'] + train_data['BsmtUnfSF']

'''Feature engineering - Binning variables'''
train_data['NeighRich'] = ''
for i in train_data['Id']:
    if train_data.at[(i-1),'Neighborhood'] in ('StoneBr', 'NridgHt', 'NoRidge'):
        train_data.at[(i-1), 'NeighRich'] = 3
        
    elif train_data.at[(i-1),'Neighborhood'] in ('SawyerW', 'NWAmes', 'Gilbert', 'Blmngtn', 'CollgCr', 'Crawfor','ClearCr', 'Somerst', 'Veenker', 'Timber' ):
        train_data.at[(i-1), 'NeighRich'] = 2

    elif train_data.at[(i-1),'Neighborhood'] in ('BrkSide','Edwards','Old Town', 'Sawyer', 'Blueste', 'SWISU', 'NPkVill', 'NAmes', 'Mitchel'):
        train_data.at[(i-1), 'NeighRich'] = 1

    else:
        train_data.at[(i-1), 'NeighRich'] = 0
'''First drop of variables'''
first_feat_drop = ['FullBath', 'BsmtFullBath', 'HalfBath', 'BsmtHalfBath', 'YearRemodAdd']

#For features I might add back in that will likely increase prediction power.
contro_first_drop = ['YrSold', 'YearBuilt', '2ndFlrSF', 'LowQualFinSF', 'BsmtFinSF2', 'BsmtUnfSF', '3SsnPorch', 'OpenPorchSF', 'EnclosedPorch', 'ScreenPorch', 'Neighborhood']
#Now drop highly correlated features:
high_corr = ['GarageCond', 'GarageArea', 'GarageYrBlt', 'TotRmsAbvGrd']

#Create target variables:
Y = pd.to_numeric(train_data['SalePrice'][0:1458], errors = 'raise')
target = ['SalePrice', 'Id']

train_data.drop(first_feat_drop + contro_first_drop + high_corr + target , axis = 1, inplace = True)

'''
df_test_corr =  train_data.drop('SalePrice', axis = 1).corr().abs().unstack().sort_values(kind="quicksort", ascending=False).reset_index()
df_test_corr.rename(columns={"level_0": "Feature 1", "level_1": "Feature 2", 0: 'Correlation Coefficient'}, inplace=True)
df_test_corr.drop(df_test_corr.iloc[1::2].index, inplace=True)
df_test_corr_nd = df_test_corr.drop(df_test_corr[df_test_corr['Correlation Coefficient'] == 1.0].index)

corr = df_test_corr_nd['Correlation Coefficient'] > 0.5
print(df_test_corr_nd[corr])
'''

'''Pre-process the data'''
#List the continuous features:
cont_feats = ['LotFrontage', 'LotArea', 'MasVnrArea', 'BsmtFinSF1', 'TotalBsmtSF', '1stFlrSF', 'GrLivArea', 'WoodDeckSF', 'PoolArea', 'MiscVal', 'Age', 'TotalSF', 'TotalPorchSF']
#List all the ordinal categorical variables:
ord_feats = ['Street', 'LotShape', 'LandSlope', 'OverallQual', 'OverallCond','MasVnrType','ExterQual', 'ExterCond','BsmtQual','BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'HeatingQC', 'CentralAir', 'BedroomAbvGr', 'KitchenAbvGr', 'KitchenQual', 'Functional', 'Fireplaces', 'FireplaceQu', 'GarageFinish', 'GarageCars', 'GarageQual', 'PavedDrive', 'PoolQC', 'TotalBath', 'Remodeled', 'IsNew', 'NeighRich']
#List of all the OHE categorical variables:
OHE_feats = ['MSSubClass', 'MSZoning', 'Alley', 'LandContour', 'LotConfig', 'Condition1', 'Condition2', 'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd', 'Foundation', 'Electrical', 'Heating', 'GarageType', 'Fence', 'MiscFeature', 'MoSold', 'SaleType', 'SaleCondition']

#Now standadize the features in the ordinal and continuous features
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

TRAIN = train_data[0:1458]
TRAIN = TRAIN.reset_index(drop = True)
TEST = train_data[1458:]

scaler = StandardScaler()

TRAIN[cont_feats] = scaler.fit_transform(TRAIN[cont_feats])
enc = OneHotEncoder(sparse = False, handle_unknown='ignore')
Ohe = pd.DataFrame(enc.fit_transform(TRAIN[OHE_feats]), columns = enc.get_feature_names())
TRAIN.drop(OHE_feats,inplace = True,  axis = 1)
Train =  pd.concat([TRAIN, Ohe], axis  = 1)

TEST[cont_feats] = scaler.transform(TEST[cont_feats])
Ohe_test = pd.DataFrame(enc.transform(TEST[OHE_feats]), columns = enc.get_feature_names())
TEST.drop(OHE_feats, inplace = True, axis = 1)
TEST = TEST.reset_index(drop = True)
Test = pd.concat([TEST, Ohe_test], axis = 1)

'''Final check
print('----------------Test data--------')
print(Test.describe())
print('-------Train data--------')
print(Train.describe())
print('--------y values-------')
print(Y.describe)
'''

#print("Skewness: %f" % Train['LotArea'].skew())
Y_train = np.log(Y)

'''
skewed = []
for i in cont_feats + ord_feats:
    if Train[i].skew() > 1.0:
        if i == 'IsNew':
            pass
        else:
            skewed.append(i)
    else:
        pass

#print(skewed)
'
log_ = ['LotFrontage', 'MasVnrArea', 'WoodDeckSF', 'TotalPorchSF']
p_1 = ['LotArea', 'GrLivArea', 'PoolArea', 'MiscVal', 'ExterCond', 'BsmtExposure', 'BsmtFinType2', 'KitchenAbvGr', 'PoolQC']

for i in log_:
    Train[i] = np.log(Train[i])
    Test[i] = np.log(Test[i])

for i in p_1:
    Train[i] = np.log1p(Train[i])
    Test[i] = np.log1p(Test[i])


    #print("Skewness: %f" % Train[i].skew())
    #print("Skewness 1p: %f" % (np.log1p(Train[i])).skew())
print("Skewness: %f" % Train['BsmtExposure'].skew())
print("Skewness: %f" % Train['BsmtFinType2'].skew())
print("Skewness: %f" % Train['KitchenAbvGr'].skew())
print("Skewness: %f" % Train['PoolQC'].skew())
'''
#print(Test.isnull().values.sum())
#print(Train.isnull().values.sum())

'''Lasso optimisation
lasso = Lasso()
parameters = {'alpha':np.linspace(0.001, 0.11, 1000)}
lasso_regressor = GridSearchCV(lasso, parameters, scoring = 'neg_mean_squared_error', cv = 5, n_jobs = -1, verbose = 1, refit = True)

lasso_regressor.fit(Train, Y_train)

print(lasso_regressor.best_params_)
print(lasso_regressor.best_score_)
print(lasso_regressor.scorer_)
'''

'''Lasso final use
reg_lasso = sklearn.linear_model.Lasso(alpha = 0.001)
X_train, X_test, Y_tr, Y_test = sklearn.model_selection.train_test_split(Train, Y_train, test_size = 0.15, random_state = 0)
reg_lasso.fit(X_train, Y_tr)
err_lasso =  sklearn.metrics.mean_squared_error(Y_test, reg_lasso.predict(X_test))
print(err_lasso)
'''
'''Ridge optomisation
ridge = Ridge()
parameters = {'alpha':np.linspace(0.001, 1, 1000)}
ridge_regressor =  GridSearchCV(ridge, parameters, scoring = 'neg_mean_squared_error', cv = 5, n_jobs = -1, verbose = 1, refit = True)
ridge_regressor.fit(Train, Y_train)
print(ridge_regressor.best_params_)
print(ridge_regressor.best_score_)
'''
'''Ridge final use
X_train, X_test, Y_tr, Y_test = sklearn.model_selection.train_test_split(Train, Y_train, test_size = 0.15, random_state = 0)
reg_ridge = Ridge(alpha = 1.0)
reg_ridge.fit(X_train, Y_tr)
err_ridge =  sklearn.metrics.mean_squared_error(Y_test, reg_ridge.predict(X_test))
print(err_ridge)
'''

'''Basic random forest regressor for XGBoost baseline
X_train, X_test, Y_tr, Y_test = sklearn.model_selection.train_test_split(Train, Y_train, test_size = 0.15, random_state = 0)
reg_forest = sklearn.ensemble.RandomForestRegressor(n_estimators = 10000, n_jobs = -1, verbose = 5)
reg_forest.fit(X_train, Y_tr)
err_forest = sklearn.metrics.mean_squared_error(Y_test, reg_forest.predict(X_test))
feature_importances = pd.DataFrame(reg_forest.feature_importances_, index = X_train.columns, columns=['importance']).sort_values('importance', ascending=False)
print(feature_importances)
print(err_forest)
'''
'''combined models
X_train, X_test, Y_tr, Y_test = sklearn.model_selection.train_test_split(Train, Y_train, test_size = 0.15)
reg_forest = sklearn.ensemble.RandomForestRegressor(n_estimators = 10000, n_jobs = -1, verbose = 5)
reg_forest.fit(X_train, Y_tr)
reg_lasso = sklearn.linear_model.Lasso(alpha = 0.001)
reg_lasso.fit(X_train, Y_tr)

las_prd = reg_lasso.predict(X_test)
forest_prd = reg_forest.predict(X_test)
combo_prd = ((2*las_prd + forest_prd)/3)
print(combo_prd.shape)
err_combo = sklearn.metrics.mean_squared_error(Y_test, combo_prd)
print(err_combo)
'''

'''XGBoost - optomisation'''
Train['IsNew'] = Train['IsNew'].astype('int64')
Train['NeighRich'] = Train['NeighRich'].astype('int64')
Train['Remodeled'] = Train['Remodeled'].astype('int64')

Test['IsNew'] = Test['IsNew'].astype('int64')
Test['NeighRich'] = Test['NeighRich'].astype('int64')
Test['Remodeled'] = Test['Remodeled'].astype('int64')



'''
xgb1 = xgb.XGBRegressor(learning_rate = 0.01, n_estimators = 10000, max_depth = 5, min_child_weight = 3, gamma = 0, subsample = 0.9, colsample_bytree = 0.43, reg_alpha = 0.0003, reg_lambda = 1, objective = 'reg:squarederror') #, verbosity = 2)
#modelfit(xgb1, Train, Y_train)


X_train, X_test, Y_tr, Y_test = sklearn.model_selection.train_test_split(Train, Y_train, test_size = 0.20, random_state = 0)
xgb1.fit(X_train, Y_tr)
xgb_prd = xgb1.predict(X_test)
err_xgb = sklearn.metrics.mean_squared_error(Y_test, xgb_prd)
print(err_xgb)
'''
'''Max_depth and min_child_weight

param_test1 = {
 'max_depth':[3,4,5],
 'min_child_weight':[1,2,3]
}

gsearch1 = GridSearchCV(estimator = xgb.XGBRegressor( learning_rate =0.01, n_estimators=2825, max_depth=4, min_child_weight=2, gamma=0, subsample=0.8, colsample_bytree=0.8, objective= 'reg:squarederror', scale_pos_weight=1, seed=0),
 param_grid = param_test1, scoring='neg_mean_squared_error',n_jobs=-1,iid=False, cv=5, verbose = 5)

gsearch1.fit(Train, Y_train)
print(gsearch1.best_score_)
print(gsearch1.best_params_)
#print(gsearch1.cv_results_)
#print(gsearch1.best_index_)
'''
'''Gamma
param_test3 = {'gamma':[i/10.0 for i in range(0,11)]}

gsearch3 = GridSearchCV(estimator = xgb.XGBRegressor( learning_rate =0.1, n_estimators=316, max_depth=4, min_child_weight=2, gamma=0, subsample=0.8, colsample_bytree=0.8, objective= 'reg:squarederror', scale_pos_weight=1, seed=0),
 param_grid = param_test3, scoring='neg_mean_squared_error',n_jobs=-1,iid=False, cv=5, verbose = 5)

gsearch3.fit(Train, Y_train)
print(gsearch3.best_score_)
print(gsearch3.best_params_)
print(gsearch3.cv_results_)
print(gsearch3.best_index_)
'''
'''Sample and colsample
param_test4 = {
        'subsample':[i/100.0 for i in np.linspace(84, 95, 12)],
        'colsample_bytree':[i/100.0 for i in np.linspace(34, 45, 12)]
}


gsearch4 = GridSearchCV(estimator = xgb.XGBRegressor( learning_rate =0.1, n_estimators=316, max_depth=4, min_child_weight=2, gamma=0, subsample=0.9, colsample_bytree=0.4, objective= 'reg:squarederror', scale_pos_weight=1, seed=0),
 param_grid = param_test4, scoring='neg_mean_squared_error',n_jobs=-1,iid=False, cv=5, verbose = 5)

gsearch4.fit(Train, Y_train)
print(gsearch4.best_score_)
print(gsearch4.best_params_)
print(gsearch4.cv_results_)
print(gsearch4.best_index_)
'''

'''Regularisation terms
param_test5 = {
        'reg_alpha':[i/100000.0 for i in np.linspace(25,35,11)],
        'reg_lambda':[i/10.0 for i in np.linspace(5,15,11)]
}
#print(param_test5)

gsearch5 = GridSearchCV(estimator = xgb.XGBRegressor( learning_rate =0.1, n_estimators=316, max_depth=4, min_child_weight=2, gamma=0, subsample=0.9, colsample_bytree=0.43, objective= 'reg:squarederror', scale_pos_weight=1, seed=0),
 param_grid = param_test5, scoring='neg_mean_squared_error',n_jobs=-1,iid=False, cv=5, verbose = 5)

gsearch5.fit(Train, Y_train)
print(gsearch5.best_score_)
print(gsearch5.best_params_)
#print(gsearch5.cv_results_)
print(gsearch5.best_index_)
'''
'''Final combo'''
reg_lasso = sklearn.linear_model.Lasso(alpha = 0.001)
#X_train, X_test, Y_tr, Y_test = sklearn.model_selection.train_test_split(Train, Y_train, test_size = 0.15, random_state = 0)
reg_lasso.fit(Train, Y_train)
las_prd = reg_lasso.predict(Test)
las_price = np.exp(las_prd)
xgb1 = xgb.XGBRegressor(learning_rate = 0.01, n_estimators = 10000, max_depth = 5, min_child_weight = 3, gamma = 0, subsample = 0.9, colsample_bytree = 0.43, reg_alpha = 0.0003, reg_lambda = 1, objective = 'reg:squarederror') #, verbosity = 2)
xgb1.fit(Train, Y_train)
xgb_prd =xgb1.predict(Test)
xgb_price = np.exp(xgb_prd)

combo_prd = ((2*(las_price) + xgb_price)/3)

d = {'Id': test_labels,
    'SalePrice': combo_prd}
submission_cv = pd.DataFrame(d, columns = ['Id','SalePrice'])
print(submission_cv.head())
submission_cv.to_csv('submission.csv', index = False)

