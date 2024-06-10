from sklearn.base import BaseEstimator, TransformerMixin

class RatingsEncoder(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X.Ratings = [0 if item == "__label__1" else 1 for item in X.Ratings]
        return X