import numpy as np
import pandas as pd
import scipy
from tqdm.notebook import tqdm

from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.metrics import ndcg_score
from math import sqrt

from src.metrics import precision_special, average_precision, argsort_top_n, mean_reciprocal_rank

class EvaluationPipeline:
    def __init__(self,
                 total_rating_data,
                 train_test_split: float = 0.2):
        self.total_rating_data = total_rating_data
        self.train_test_split = train_test_split
        self.test_data = self.get_test_data(self.total_rating_data)
        self.train_data = self.total_rating_data[~self.total_rating_data.index.isin(self.test_data.index)]

    def get_test_data(self, total_data):
        return total_data.groupby('UserID', group_keys=False).apply(
            lambda x: x.tail(int(np.round(x.shape[0]*self.train_test_split))))

    def evaluate(self,
                 model_object,
                 metrics_list = False,
                 user_average_metrics: bool = False,
                 retrain_model_each_point: bool = False):
        if not metrics_list:
            metrics_list = ['mae','rmse','precision','average_precision',
                            'mean_reciprocal_rank','ndcg','coverage']
        if user_average_metrics:
            recommendation_results = {}
            ratings_y_pred = {}
        else:
            recommendation_results = []
            ratings_y_pred = []
        sorted_total_data_test = self.total_rating_data.sort_values('Timestamp')
        rows_before_timestamp_test = sorted_total_data_test.groupby('Timestamp').count()[
            'UserID'].cumsum().shift(1).fillna(0).astype(int).to_dict()
        if retrain_model_each_point:
            sorted_data_train = self.train_data.sort_values('Timestamp')
            rows_before_timestamp_train = sorted_data_train.groupby('Timestamp').count()[
                'UserID'].cumsum().shift(1).fillna(0).astype(int)
            timestamps_not_in_train = [timestamp for timestamp in rows_before_timestamp_test.keys(
                ) if timestamp not in rows_before_timestamp_train.index]
            rows_before_timestamp_train = pd.concat([rows_before_timestamp_train,
                                                     pd.Series(None,
                                                               index=timestamps_not_in_train)])
            rows_before_timestamp_train = rows_before_timestamp_train.sort_index().ffill().astype(int)
        for i_p, test_point in tqdm(self.test_data.iterrows(),
                                    total=self.test_data.shape[0]):
            test_point_timestamp = test_point['Timestamp']
            test_point_user = test_point['UserID']
            if user_average_metrics and test_point_user not in recommendation_results.keys():
                recommendation_results[test_point_user] = []
                ratings_y_pred[test_point_user] = []
            if retrain_model_each_point:
                train_data_each_point = sorted_data_train[:rows_before_timestamp_train[test_point_timestamp]]
                model_object.fit(train_data_each_point)
            # Using the whole dataset, as for each new test point for the user previous test points matter too
            items_pred, ratings_pred = model_object.predict(sorted_total_data_test[
                                                                :rows_before_timestamp_test[test_point_timestamp]],
                                                            test_point_user,
                                                            test_point_timestamp)
            if user_average_metrics:
                recommendation_results[test_point_user].append([i_p, items_pred])
                ratings_y_pred[test_point_user].append(ratings_pred[items_pred.tolist().index(
                    test_point_user)] if (test_point_user in items_pred) else 0)
            else:
                recommendation_results.append([i_p, items_pred])
                ratings_y_pred.append(ratings_pred[items_pred.tolist().index(
                    test_point_user)] if (test_point_user in items_pred) else 0)
        metrics_output_dict = {}
        if not user_average_metrics:
            ratings_y_true = [self.test_data.loc[
                              test_point_index, 'Rating'] for test_point_index in self.test_data.index]
            # ratings_y_pred = [ratings_pred[items_pred.index(
            #     self.test_data.loc[test_point_index, 'movieID'])] if self.test_data.loc[
            #         test_point_index, 'movieID'] in items_pred else 0 for test_point_index in self.test_data.index]
            # ratings_y_pred = [recommendation_results[i_p][2][recommendation_results[i_p][1].tolist().index(
            #     self.test_data.loc[test_point_index, 'MovieID'])] if self.test_data.loc[
            #         test_point_index, 'MovieID'] in recommendation_results[i_p][
            #             1] else 0 for i_p, test_point_index in enumerate(self.test_data.index)]
            self.test_data['Rating pred'] = ratings_y_pred
            ratings_y_true_users = self.test_data.groupby('UserID')['Rating'].apply(list).to_dict()
            ratings_y_pred_users = self.test_data.groupby('UserID')['Rating pred'].apply(list).to_dict()
            largest_user_id_total = self.total_rating_data['UserID'].max()
            items_id_pred = [pred[1][0] if len(pred[1]) > 0 else (
                largest_user_id_total + 1) for pred in recommendation_results]
            for metric in metrics_list:
                if metric == 'mae':
                    metrics_output_dict['mae'] = mean_absolute_error(ratings_y_true, ratings_y_pred)
                elif metric == 'rmse':
                    metrics_output_dict['rmse'] = sqrt(mean_squared_error(ratings_y_true, ratings_y_pred))
                elif metric == 'precision':
                    metrics_output_dict['precision'] = precision_special(self.test_data['MovieID'].to_numpy(),
                                                                         items_id_pred)
                elif metric == 'average_precision':
                    average_precision_list = []
                    for user in self.test_data['UserID'].unique():
                        m_user = len(ratings_y_true_users[user])
                        average_precision_list.append(average_precision(
                            argsort_top_n(ratings_y_true_users[user], m_user),
                            argsort_top_n(ratings_y_pred_users[user], m_user),
                            m_user))
                    metrics_output_dict['average_precision'] = np.mean(average_precision_list)
                elif metric == 'mean_reciprocal_rank':
                    metrics_output_dict['mean_reciprocal_rank'] = mean_reciprocal_rank(
                        self.test_data['MovieID'].to_numpy(),
                        [pred[1] for pred in recommendation_results])
                elif metric == 'ndcg':
                    ndcg = []
                    for user in self.test_data['UserID'].unique():
                        m_user = len(ratings_y_true_users[user])
                        if m_user > 1:
                            # NDCG only is defined is there is more than 1 point
                            ndcg.append(ndcg_score(
                                np.array(ratings_y_true_users[user])[
                                    argsort_top_n(ratings_y_true_users[user], m_user)].reshape(1,-1),
                                np.array(ratings_y_pred_users[user])[
                                    argsort_top_n(ratings_y_pred_users[user], m_user)].reshape(1,-1)))
                    if len(ndcg) > 0:
                        metrics_output_dict['ndcg'] = np.mean(ndcg)
                    else:
                        metrics_output_dict['ndcg'] = 0.0
                elif metric == 'coverage':
                    items_train_unique = self.train_data['UserID'].unique()
                    metrics_output_dict['coverage'] = len(np.unique([
                        item for item in items_id_pred if item in items_train_unique]))/len(items_train_unique)
        return metrics_output_dict # recommendation_results
