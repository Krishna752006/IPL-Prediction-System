import numpy as np


class IPLModelBundle:
    def __init__(self, model, dataset_version: str, feature_version: str):
        self.model = model
        self.dataset_version = dataset_version
        self.feature_version = feature_version

    def preprocess(self, X: np.ndarray, seq_len: int = 30):
        if len(X.shape) != 2:
            raise ValueError("Expected 2D feature matrix")
            
        # Pad with zeros if the inning has fewer balls than seq_len
        if X.shape[0] < seq_len:
            pad = np.zeros((seq_len - X.shape[0], X.shape[1]), dtype=np.float32)
            X = np.vstack([pad, X])
        else:
            # Crop to the most recent balls if it exceeds seq_len
            X = X[-seq_len:]
            
        return X.reshape(1, seq_len, X.shape[1])

    def predict(self, X: np.ndarray):
        """
        X should already be processed features
        """

        X = self.preprocess(X)
        preds = self.model.predict(X)

        return preds

    def info(self):
        return {
            "dataset_version": self.dataset_version,
            "feature_version": self.feature_version,
        }
