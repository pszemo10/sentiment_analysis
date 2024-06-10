import pandas as pd
from sklearn.pipeline import Pipeline
from ColumnSeparator import ColumnSeparator
from RatingsEncoder import RatingsEncoder
from MyTokenizer import MyTokenizer
from SequencePadder import SequencePadder

class Dataloader:
    def __init__(self, tokenizer_filename, tokenizer_from_file):
        self.pipe = Pipeline([
            ("separator", ColumnSeparator()),
            ("ratings encoder", RatingsEncoder()),
            ("tokenizer", MyTokenizer(tokenizer_filename, tokenizer_from_file)),
            ("padder", SequencePadder())
        ])
    
    def load(self, filename):
        data = pd.read_csv(filename, sep="\t", names=["Reviews"])
        # data = data.head(10000)
        print("Read data from file.")
        return self.pipe.fit_transform(data)
    
    def vocab_size(self):
        return self.pipe["tokenizer"].vocab_size()
        