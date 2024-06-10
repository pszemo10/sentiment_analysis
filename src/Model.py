from keras.models import Sequential
from keras.layers import Dense, Embedding, GlobalAveragePooling1D

class Model(Sequential):
    def __init__(self, vocab_size):
        super().__init__()
        self.add(Embedding(vocab_size, 128))
        self.add(GlobalAveragePooling1D())
        self.add(Dense(512, activation="relu"))
        self.add(Dense(512, activation="relu"))
        self.add(Dense(1, activation="sigmoid"))

        self.summary()
        self.compile(loss="binary_crossentropy",optimizer="adam", metrics=["accuracy"])
