import numpy as np
import pandas as pd

from src.models.abstract_rs_model import AbstractRSModel

class BaselineMeanRatingModel(AbstractRSModel):
    def __init__(self):
        self.pre_fit = False
    
    def fit(self, train_data, pre_fit: bool = False):
        if self.pre_fit:
            # The train data was already pre-fit
            self.mean_ratings = train_data[['MovieID','Rating']].groupby('MovieID')['Rating'].mean()
        else:
            self.mean_ratings = train_data[['MovieID','Rating']].groupby('MovieID')['Rating'].mean()
        self.pre_fit = pre_fit

    def predict(self, data_at_test_timestamp, test_user, test_timestamp):
        mean_ratings_candidates = self.mean_ratings[~self.mean_ratings.index.isin(
            data_at_test_timestamp[data_at_test_timestamp['UserID'] == test_user]['MovieID'].unique())]
        mean_ratings_candidates = mean_ratings_candidates.sort_values(ascending=False) # kind='mergesort'
        return mean_ratings_candidates.index.to_numpy(), mean_ratings_candidates.to_numpy()

    def fit_predict(self, data, test_user, test_timestamp):
        self.fit(data)
        return self.predict(data, test_user, test_timestamp)
