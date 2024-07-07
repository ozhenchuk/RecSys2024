from abc import ABC, abstractmethod

class AbstractRSModel:
    def __init__(self):
        pass

    @abstractmethod
    def fit(self, train_data):
        ...

    @abstractmethod
    def predict(self, data_at_test_timestamp, test_user, test_timestamp):
        ...
        # Returns: list of predicted items, list of their predicted ratings
