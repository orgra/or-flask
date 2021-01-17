from joblib import load
from sklearn.ensemble import RandomForestClassifier
import numpy as np


class Model:

    def __init__(self, path_to_model):
        self.model = load(path_to_model)

    def predicting(self, data):
        data_formatted = np.ndarray(data)
        return self.model.predict(data_formatted)
