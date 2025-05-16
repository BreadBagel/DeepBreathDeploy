import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression

class LogisticFusionModel:
    """
    Logistic Regression fusion model combining ResNet output probability and symptom data.
    """
    def __init__(self, model_path: str = None):
        """
        Initialize the fusion model. If model_path is provided, load the saved model.
        """
        self.model = None
        if model_path:
            self.load(model_path)

    def train(self, X: np.ndarray, y: np.ndarray, save_path: str = None):
        """
        Train the logistic regression model on feature matrix X and labels y.
        Optionally save the trained model to disk.

        Parameters:
            X (np.ndarray): shape (n_samples, n_features)
            y (np.ndarray): shape (n_samples,)
            save_path (str): filepath to save the trained model
        """
        clf = LogisticRegression(solver='liblinear')
        clf.fit(X, y)
        self.model = clf
        if save_path:
            joblib.dump(self.model, save_path)
        return clf

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        """
        Return probability estimates for the positive class.
        """
        if self.model is None:
            raise ValueError("Logistic model not trained or loaded.")
        arr = np.array(features)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        return self.model.predict_proba(arr)[:, 1]

    def predict(self, features: np.ndarray) -> np.ndarray:
        """
        Return binary class predictions (0 or 1) for given features.
        """
        if self.model is None:
            raise ValueError("Logistic model not trained or loaded.")
        arr = np.array(features)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        return self.model.predict(arr)

    def save(self, path: str):
        """
        Save the trained logistic regression model to the specified path.
        """
        if self.model is None:
            raise ValueError("No logistic model to save.")
        joblib.dump(self.model, path)

    def load(self, path: str):
        """
        Load a saved logistic regression model from disk.
        """
        self.model = joblib.load(path)
        return self.model
