import pandas as pd

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
test['SalePrice'] =  'NA'
train_data = pd.concat([train, test], ignore_index=True)

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
        for i in train_data.loc[(train_data[col_list[0]]).isnull() & (train_data[col_list[1]].isnull()) & (train_data[col_list[2]].isnull() )& (train_data[col_list[3]].isnull())]['Id']:
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
            print(dfCMR_[(train_data[list_var[1]][(i-1)]),0])
    
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
    return





def print_col(col_name):
    z= print(train_data[col_name])

    return



if __name__ == '__main__':
    cat = ['BsmtQual']
    value = [0]
    cat_mode_replace(cat,value)
