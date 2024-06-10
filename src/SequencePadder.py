from sklearn.base import BaseEstimator, TransformerMixin
from keras.utils import pad_sequences

class SequencePadder(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        padded = pad_sequences(X.Reviews, maxlen=100, truncating="post", padding="post")
        X.Reviews = [el for el in padded]
        return X