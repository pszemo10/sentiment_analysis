from Dataloader import Dataloader
from Model import Model
import tensorflow as tf
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.callbacks import History, ModelCheckpoint
from sklearn.model_selection import train_test_split
import os 

TRAINING_FILE = os.path.join("..", "data", "train.ft.txt")
MODEL_DIR = os.path.join("..", "model")
TOKENIZER_FILE = "tokenizer.json"
MODEL_FILE = "model_weights.h5py"

def main():
    dataloader = Dataloader(os.path.join(MODEL_DIR, TOKENIZER_FILE), False)
    df = dataloader.load(TRAINING_FILE)
    print(df['Ratings'].value_counts())

    X = df['Reviews']
    y = df['Ratings']
    
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, stratify=y)
    
    
    X_train = np.array([np.asarray(el).astype('int32') for el in X_train])
    X_val = np.array([np.asarray(el).astype('int32') for el in X_val])
    y_train = np.asarray(y_train).astype('int32').reshape((-1,1))
    y_val = np.asarray(y_val).astype('int32').reshape((-1,1))
    
    history = History()
    save_best_model = ModelCheckpoint(os.path.join(MODEL_DIR, MODEL_FILE),save_best_only=True)
    model = Model(dataloader.vocab_size()+1)
    model.fit(X_train, y_train, validation_data=(X_val, y_val), batch_size=32,epochs=2, callbacks=[history, save_best_model])

if __name__ == "__main__":
    main()