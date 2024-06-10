from sklearn.base import BaseEstimator, TransformerMixin
from keras.preprocessing.text import tokenizer_from_json
from keras.preprocessing.text import Tokenizer
import json

class MyTokenizer(BaseEstimator, TransformerMixin):
    def __init__(self, filename, read_from_file):
        self.read_from_file = read_from_file
        self.filename = filename
        if read_from_file:
            with open(filename) as file:
                data = json.load(file)
                self.tokenizer = tokenizer_from_json(data)
        else:
            self.tokenizer = Tokenizer(oov_token = "oov token")

    def fit(self, X, y=None):
        if not self.read_from_file:
            self.tokenizer.fit_on_texts(X.Reviews)
            with open(self.filename, 'w', encoding='utf-8') as file:
                file.write(json.dumps(self.tokenizer.to_json(), ensure_ascii=False))
        return self
    
    def transform(self, X):
        X.Reviews = self.tokenizer.texts_to_sequences(X.Reviews)
        return X
    
    def vocab_size(self):
        return len(self.tokenizer.word_index)