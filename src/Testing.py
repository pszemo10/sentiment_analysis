from Dataloader import Dataloader
from Model import Model
import tensorflow as tf
import numpy as np
import os 

TRAINING_FILE = os.path.join("..", "data", "test.ft.txt")
MODEL_DIR = os.path.join("..", "model")
TOKENIZER_FILE = "tokenizer.json"
MODEL_FILE = "model_weights.h5py"

def main():
    dataloader = Dataloader(os.path.join(MODEL_DIR, TOKENIZER_FILE), True)
    df = dataloader.load(TRAINING_FILE)
    
    X_test = df['Reviews']
    y_test = df['Ratings']

    X_test = np.array([np.asarray(el).astype('int32') for el in X_test])
    y_test = np.asarray(y_test).astype('int32').reshape((-1,1))

    model = Model(dataloader.vocab_size()+1)
    model.load_weights(os.path.join(MODEL_DIR, MODEL_FILE))
    model.evaluate(X_test,y_test)


if __name__ == "__main__":
    main()