import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import math
from collections import Counter
from sklearn import preprocessing

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
test['SalesPrice'] =  'NA'
train_data = pd.concat([train, test], ignore_index=True)

def cat_frequency(category):
    '''Breakdown of the frequency of occurence of each value in a categorical feature.
    '''
    df = train_data.groupby([category]).sum()
    counts = np.zeros(len(df.index))
    cat = []
    for i,j in zip(df.index,range(len(df.index))):
        value = (train_data[category]==i).sum()
        counts[j] = value
        cat.append(i)
    cats = np.asarray(cat)
    sns.barplot(x=cats, y=counts)
    plt.xlabel(category, fontsize=15)
    plt.ylabel('Counts', fontsize=15)
    plt.show()

    return

def bar_graph_cats(cats):
    '''Plots the values that occur in a feature with respect to another categorical feature, both as bar graphs. The first string in the list of categories is the main one to be broken down.
    The second string is the feature to be plotted with. '''
    df = train_data[cats]
    df1 = df.groupby(cats[1])
    dfg1 = df1[cats[0]].value_counts(normalize=True)
    dfg1.unstack().plot(kind='bar')
    plt.xticks(rotation=45)
    plt.show()

    return

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


if __name__ == '__main__':
    cats = ['BsmtQual', 'Neighborhood']
    bar_graph_cats(cats)

    cat = 'Neighborhood'
    cat_frequency(cat)
