import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn import preprocessing

train_data = pd.read_csv('train.csv')
'''
print(len(train_data))
total2 = train_data.select_dtypes(include=['int','float']).isnull().sum().sort_values(ascending=False)
total = train_data.isnull().sum().sort_values(ascending=False)
percent = (train_data.isnull().sum()/train_data.isnull().count()).sort_values(ascending=False)
percent2 = (train_data.select_dtypes(include=['int','float']).isnull().sum()/train_data.select_dtypes(include=['int','float']).isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total missing', 'Percent missing'])
missing_cont = pd.concat([total2, percent2], axis=1, keys=['Total missing', 'Percent missing'])
print(missing_cont.info())
f, ax = plt.subplots(figsize=(15, 6))
plt.xticks(rotation='90')
sns.barplot(x=missing_cont.index, y=missing_cont['Percent missing'])
plt.xlabel('Features', fontsize=15)
plt.ylabel('Percent of missing values', fontsize=15)
plt.title('Percent missing continuous data by feature', fontsize=15)
plt.show()
print(missing_data.head(81))
'''



'''
df = train_data.groupby(['Neighborhood']).sum()
counts = np.zeros(len(df.index))
cat = []
#print(cat.shape)
for i,j in zip(df.index,range(len(df.index))):
    value = (train_data['Neighborhood']==i).sum()
    counts[j] = value
    cat.append(i)
cats = np.asarray(cat)
print(len(counts))
print(len(cats))
'''



df = train_data[['Neighborhood','BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinSF1','BsmtFinType2','BsmtFinSF2','BsmtUnfSF','TotalBsmtSF','BsmtFullBath','BsmtHalfBath']]
df1 = df.groupby(['Neighborhood'])
dfg1 = df1['BsmtFinType2'].value_counts()
print(dfg1.head(10))
dfg1.unstack().plot(kind='bar')
plt.xticks(rotation=45)
plt.show()

df2 = df.groupby(['BsmtQual'])
dfg2 = df2['BsmtFinType2'].value_counts()
dfg2.unstack().plot(kind='bar')
plt.xticks(rotation=45)
plt.show()

df3 = df.groupby(['BsmtCond'])
dfg3 = df3['BsmtFinType2'].value_counts()
dfg3.unstack().plot(kind='bar')
plt.xticks(rotation=45)
plt.show()

df4 = df.groupby(['BsmtExposure'])
dfg4 = df4['BsmtFinType2'].value_counts()
dfg4.unstack().plot(kind='bar')
plt.xticks(rotation=45)
plt.show()


df5 = df.groupby(['BsmtFinType1'])
dfg5 = df5['BsmtFinType2'].value_counts()
dfg5.unstack().plot(kind='bar')
plt.xticks(rotation=45)
plt.show()
'''
print('---------------------------------------------------------------------------------')
print('---------------------------------------------------------------------------------')
print('---------------------------------------------------------------------------------')

t = (train_data['GarageType']=='2Types').sum()
Attc = (train_data['GarageType']=='Attchd').sum()
Bsm = (train_data['GarageType']=='Basment').sum()
Bin = (train_data['GarageType']=='BuiltIn').sum()
Cpt = (train_data['GarageType']=='CarPort').sum()
Dtd = (train_data['GarageType']=='Detchd').sum()
NA = (train_data['GarageType']=='NA').sum()
Miss_ga = (train_data['GarageType'].isnull().sum())
Tot_ga =  (t + Attc + Bsm + Bin + Cpt + Dtd + NA + Miss_ga)

print('Number of 2 type garages ='+str(t))
print('Number of attached garages ='+str(Attc))
print('Number of basement garages ='+str(Bsm))
print('Number of built in garages ='+str(Bin))
print('Number of car port garages ='+str(Cpt))
print('Number of detached garages ='+str(Dtd))
print('No garages ='+str(NA))
print('Missing values ='+str(Miss_ga))
print('The total number of counts is ='+str(Tot_ga))

#Counts2 = np.array([t, Attc, Bsm, Bin, Cpt, Dtd, NA, Miss_ga])
#Garages = ['Two garages', 'Attched', 'Basement', 'Built in', 'Car port', 'Detached', 'NA']
#sns.barplot(x=Garages, y=Counts2)
#plt.xlabel('Type of Garage', fontsize=15)
#plt.ylabel('Counts', fontsize=15)
#plt.title('Breakdown of garages', fontsize=15)
#plt.show()


print('---------------------------------------------------------------------------------')
print('---------------------------------------------------------------------------------')
print('---------------------------------------------------------------------------------')

Ex_qual = (train_data['PoolQC']=='Ex').sum()
Gd_qual = (train_data['PoolQC']=='Gd').sum()
TA_qual = (train_data['PoolQC']=='TA').sum()
Fa_qual = (train_data['PoolQC']=='Fa').sum()
NA_qual = (train_data['PoolQC']=='Na').sum()
Miss_qual = (train_data['PoolQC'].isnull().sum())
Tot_qual =  (Ex_qual + Gd_qual + TA_qual + Fa_qual + NA_qual + Miss_qual)

print('Excellent quality ='+str(Ex_qual))
print('Good quality ='+str(Gd_qual))
print('Typical ='+str(TA_qual))
print('Fair quality ='+str(Fa_qual))
print('No basement ='+str(NA_qual))
print('Missing values ='+str(Miss_qual))
print('Total counts ='+str(Tot_qual))

Counts3 = np.array([Ex_qual, Gd_qual, TA_qual, Fa_qual,NA_qual,Miss_qual])
Quality = ['Excellent', 'Good', 'Typical', 'Fair', 'No basement', 'Missing values']
sns.barplot(x=Quality, y=Counts3)
plt.xlabel('Quality of pool', fontsize=15)
plt.ylabel('Counts', fontsize=15)
plt.title('Breakdown of pool quality', fontsize=15)
plt.show()



print('---------------------------------------------------------------------------------')
print('---------------------------------------------------------------------------------')
print('---------------------------------------------------------------------------------')


Elev = (train_data['MiscFeature']=='Elev').sum()
Gar2 = (train_data['MiscFeature']=='Gar2').sum()
Othr = (train_data['MiscFeature']=='Othr').sum()
Shed = (train_data['MiscFeature']=='Shed').sum()
TenC = (train_data['MiscFeature']=='TenC').sum()
NA_misc = (train_data['MiscFeature']=='Na').sum()
Miss = (train_data['MiscFeature'].isnull().sum())
Tot_cond =  (Elev + Gar2 + Othr + Shed + TenC + NA_misc + Miss)

print('Elevator ='+str(Elev))
print('2nd Garage ='+str(Gar2))
print('Other ='+str(Othr))
print('Shed ='+str(Shed))
print('Tennis Court ='+str(TenC))
print('None  ='+str(NA_misc))
print('Missing values ='+str(Miss))
print('Total counts ='+str(Tot_cond))

Counts4 = np.array([Elev, Gar2, Othr, Shed, TenC, NA_misc, Miss])
Features= ['Elevator', '2nd Garage', 'Other', 'Shed', 'Tennis Court', 'No features', 'Missing values']
sns.barplot(x=Features, y=Counts4)
plt.xlabel('Features', fontsize=15)
plt.ylabel('Counts', fontsize=15)
plt.title('Breakdown of Misc features', fontsize=15)
plt.show()

Maps for label encoding

Qualmap ={'Ex': 0, 'Gd':1, 'TA':2, 'Fa':3, 'Po':4}
Zonemap = {'A':0, 'C':1, 'FV':2, 'I':3, 'RH':4,'RL':5, 'RP':6, 'RM':7}
streetmap = {'Grvl':1, 'Pave':2}
Alleymap = {'Grvl':1, 'Pave':2}
lotshapemap = {'Reg':0,'IR1':1,'IR2':2,'IR3':3}
landcontmap = {'Lvl':0, 'Bnk':1, 'HLS':2, 'Low':3}
Utilitiesmap = {'AllPub':0, 'NoSewr':1, 'NoSeWa':2, 'ELO':3}
LotConfigmap= {'Inside':0, 'Corner':1, 'CulDSac':2, 'FR2':3, 'FR3':4}
LandSlopemap = {'Gtl':0, 'Mod':1, 'Sev':2}
Neighborhoodmap = {'Blmngtn':0, 'Blueste':1, 'BrDale':2, 'BrkSide':3, 'ClearCr':4, 'CollgCr':5, 'Crawfor':6, 'Edwards':7,
                  'Gilbert':8, 'IDOTRR':9, 'MeadowV':10, 'Mitchel':11, 'NAmes':12, 'NoRidge':13, 'NPkVill':14, 'NridgHt':15,
                  'NWAmes':16, 'OldTown':17, 'SWISU':18, 'Sawyer':19, 'SawyerW':20, 'Somerst':21, 'StoneBr':22, 'Timber':23, 'Veenker':24}
Conditionmap = {'Artery':0, 'Feedr':1, 'Norm':2, 'RRNn':3, 'RRAn':4, 'PosN':5, 'PosA':6, 'RRNe':7, 'RRAe':8}
BldgTypemap = {'1Fam':0,'2fmcon':1, 'Duplex':2, 'TwnhsE':3, 'TwnhsI':4}
HouseStylemap = {'1Story':0, '1.5Fin':1, '1.5Unf':2, '2Story':3, '2.5Fin':4, '2.5Unf':5, 'SFoyer':6, 'SLvl':7}
RoofStylemap = {'Flat':0, 'Gable':1, 'Gambrel':2, 'Hip':3, 'Mansard':4, 'Shed':5}
RoofMatlmap = {'ClyTile':0, 'CompShg':1, 'Membran':2, 'Metal':3, 'Roll':4, 'Tar&Grv':5, 'WdShake':6, 'WdShngl':7}
Exteriormap = {'AsbShng':0, 'AsphShn':1, 'BrkComm':2, 'BrkFace':3, 'CBlock':4, 'CemntBd':5, 'HdBoard':6, 'ImStucc':7, 'MetalSd':8, 'Other':9, 'Plywood':10, 'PreCast':11,
		'Stone':12, 'Stucco':13, 'VinylSd':14, 'Wd Sdng':15, 'WdShing':16}
MasVnrTypemap = {'BrkCmn':1, 'BrkFace':2, 'CBlock':3, 'Stone':4}
Foundationmap = {'BrkTil':0, 'CBlock':1, 'PConc':2, 'Slab':3, 'Stone':4, 'Wood':5}
BsmtQualmap = {'Ex': 1, 'Gd':2, 'TA':3, 'Fa':4, 'Po':5}
BsmtExposuremap = {'Gd':1, 'Av':2, 'Mn':3, 'No':4}
BsmtFinTypemap = {'GLQ':1, 'ALQ':2, 'BLQ':3, 'Rec':4, 'LwQ':5, 'Unf':6}
Heatingmap = {'Floor':0, 'GasA':1, 'GasW':2, 'Grav':3, 'OthW':4, 'Wall':5}
CentralAirmap = {'N':0, 'Y':1}
Electricalmap = {'SBrkr':0, 'FuseA':1, 'FuseF':2, 'FuseP':3, 'Mix':4}
Functionalmap = {'Typ':0, 'Min1':1, 'Min2':2, 'Mod':3, 'Maj1':4, 'Maj2':5, 'Sev':6, 'Sal':7}
GarageTypemap = {'2Types':1, 'Attchd':2, 'Basment':3, 'BuiltIn':4, 'CarPort':5, 'Detchd':6}
GarageFinishmap = {'Fin':1, 'RFn':2, 'Unf':3}
PavedDrivemap = {'N':0, 'Y':1, 'P':2}
Fencemap = {'GdPrv':1, 'MnPrv':2,'GdWo':3,'MnWw':4}
MiscFeaturemap = {'Elev':1, 'Gar2':2, 'Othr':3, 'Shed':4, 'TenC':5}
SaleTypemap = {'WD':0, 'CWD':1, 'VWD':2, 'New':3, 'COD':4, 'Con':5, 'ConLw':6, 'ConLI':7, 'ConLD':8, 'Oth':9}
SaleConditionmap = {'Normal':0, 'Abnorml':1, 'AdjLand':2, 'Alloca':3, 'Family':4, 'Partial':5}


train_data['MSZoning']=train_data['MSZoning'].map(Zonemap)
train_data['Street']=train_data['Street'].map(streetmap)
train_data['Alley']=train_data['Alley'].map(Alleymap)
train_data['LotShape']=train_data['LotShape'].map(lotshapemap)
train_data['LandContour']=train_data['LandContour'].map(landcontmap)
train_data['Utilities']=train_data['Utilities'].map(Utilitiesmap)
train_data['LotConfig']=train_data['LotConfig'].map(LotConfigmap)
train_data['LandSlope']=train_data['LandSlope'].map(LandSlopemap)
train_data['Neighborhood']=train_data['Neighborhood'].map(Neighborhoodmap)
train_data['Condition1']=train_data['Condition1'].map(Conditionmap)
train_data['Condition2']=train_data['Condition2'].map(Conditionmap)
train_data['BldgType']=train_data['BldgType'].map(BldgTypemap)
train_data['HouseStyle']=train_data['HouseStyle'].map(HouseStylemap)
train_data['RoofStyle']=train_data['RoofStyle'].map(RoofStylemap)
train_data['RoofMatl']=train_data['RoofMatl'].map(RoofMatlmap)
train_data['Exterior1st']=train_data['Exterior1st'].map(Exteriormap)
train_data['Exterior2nd']=train_data['Exterior2nd'].map(Exteriormap)
train_data['MasVnrType']=train_data['MasVnrType'].map(MasVnrTypemap)
train_data['ExterQual']=train_data['ExterQual'].map(Qualmap)
train_data['ExterCond']=train_data['ExterCond'].map(Qualmap)
train_data['Foundation']=train_data['Foundation'].map(Foundationmap)
train_data['BsmtQual']=train_data['BsmtQual'].map(BsmtQualmap)
train_data['BsmtCond']=train_data['BsmtCond'].map(BsmtQualmap)
train_data['BsmtExposure']=train_data['BsmtExposure'].map(BsmtExposuremap)
train_data['BsmtFinType1']=train_data['BsmtFinType1'].map(BsmtFinTypemap)
train_data['BsmtFinType2']=train_data['BsmtFinType2'].map(BsmtFinTypemap)
train_data['Heating']=train_data['Heating'].map(Heatingmap)
train_data['HeatingQC']=train_data['HeatingQC'].map(Qualmap)
train_data['CentralAir']=train_data['CentralAir'].map(CentralAirmap)
train_data['Electrical']=train_data['Electrical'].map(Electricalmap)
train_data['KitchenQual']=train_data['KitchenQual'].map(Qualmap)
train_data['Functional']=train_data['Functional'].map(Functionalmap)
train_data['FireplaceQu']=train_data['FireplaceQu'].map(BsmtQualmap)
train_data['GarageType']=train_data['GarageType'].map(GarageTypemap)
train_data['GarageFinish']=train_data['GarageFinish'].map(GarageFinishmap)
train_data['GarageQual']=train_data['GarageQual'].map(BsmtQualmap)
train_data['GarageCond']=train_data['GarageCond'].map(BsmtQualmap)
train_data['PavedDrive']=train_data['PavedDrive'].map(PavedDrivemap)
train_data['PoolQC']=train_data['PoolQC'].map(BsmtQualmap)
train_data['Fence']=train_data['Fence'].map(Fencemap)
train_data['MiscFeature']=train_data['MiscFeature'].map(MiscFeaturemap)
train_data['SaleType']=train_data['SaleType'].map(SaleTypemap)
train_data['SaleCondition']=train_data['SaleCondition'].map(SaleConditionmap)
#print(train_data['Neighborhood'].head())

None       864
BrkFace    445
Stone      128
BrkCmn      15
0            8

None       864
BrkFace    445
Stone      128
BrkCmn      15

'''
