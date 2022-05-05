""" Preprocessing : CV splits for Train set & merge the csv file for Train and Test set.
"""

import numpy as np
import pandas as pd

from sklearn import datasets
from sklearn import model_selection

from transformers import AutoTokenizer, AutoModel, AutoConfig


def create_folds(data, num_splits):
    # we create a new column called kfold and fill it with -1
    data["kfold"] = -1
    
    # the next step is to randomize the rows of the data
    data = data.sample(frac=1).reset_index(drop=True)

    # calculate number of bins by Sturge's rule
    # I take the floor of the value, you can also
    # just round it
    num_bins = int(np.floor(1 + np.log2(len(data))))
    
    # bin targets
    data.loc[:, "bins"] = pd.cut(
        data["score"], bins=num_bins, labels=False
    )
    
    # initiate the kfold class from model_selection module
    kf = model_selection.StratifiedKFold(n_splits=num_splits, shuffle=True, random_state=42)
    
    # fill the new kfold column
    # note that, instead of targets, we use bins!
    for f, (t_, v_) in enumerate(kf.split(X=data, y=data.bins.values)):
        data.loc[v_, 'kfold'] = f
    
    # drop the bins column
    data = data.drop("bins", axis=1)

    # return dataframe with folds
    return data

if __name__ == '__main__':

    # preprocess the training set.
    train_df = pd.read_csv("./data/train.csv")
    titles = pd.read_csv('./data/titles.csv')
    train_df = train_df.merge(titles, left_on='context', right_on='code')

    train_df = create_folds(train_df, num_splits=5)
    
    train_df.to_csv("./data/train_pre.csv", index=False)

    #preprocess the test set.
    test_df = pd.read_csv("./data/test.csv")
    titles = pd.read_csv('./data/titles.csv')
    test_df = test_df.merge(titles, left_on='context', right_on='code')
    test_df.to_csv("./data/test_pre.csv", index=False)