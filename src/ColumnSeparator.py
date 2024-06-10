from sklearn.base import BaseEstimator, TransformerMixin

class ColumnSeparator(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        ratings = X.applymap(lambda row: row.split()[0])
        reviews = X.applymap(lambda row:  " ".join(row.split()[1:]))
        X['Ratings'] = ratings
        X['Reviews'] = reviews
        
        return X