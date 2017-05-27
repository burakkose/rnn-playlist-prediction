import numpy as np
from data import DatasetMode, load
from keras.layers import Dense, GRU, Embedding
from keras.models import Sequential
from keras.optimizers import Adam
from keras.preprocessing import sequence
from keras.utils.np_utils import to_categorical
from metrics import top_k_accuracy_func_list


class PlaylistGeneration:
    def __init__(self, mode=DatasetMode.small):
        self.optimizer = \
            Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
        self.model = Sequential()
        self.activation = 'softmax'
        self.loss = 'categorical_crossentropy'
        self.metrics = top_k_accuracy_func_list([50, 100, 200, 300, 400, 500])

        '''
        Index of songs in x_train(or test) starts from 1 because of zero padding.
        Index of songs in y_train(or test) starts from zero like song hash.
        For instance:
        In dataset, index of songA is 21.
        songA's index is 22 in x_train(or test)
        songA's index is 21 in y_train(or test).
        The goal is the neural network having the ability to ignore zero-paddings
        '''
        (x_train, y_train), (x_test, y_test), songs = load(mode)

        self.max_length = max([len(playlist) for playlist in x_train])
        self.song_hash = songs

        self.x_train = np.asarray(sequence.pad_sequences(x_train, maxlen=self.max_length), dtype="int64")
        self.y_train = to_categorical(y_train, len(self.song_hash) + 1)  # Zero is included

        self.x_test = np.asarray(sequence.pad_sequences(x_test, maxlen=self.max_length), dtype="int64")
        self.y_test = to_categorical(y_test, len(self.song_hash) + 1)  # Zero is included

    def process(self):
        self.model.add(Embedding(len(self.song_hash) + 1, 50, dropout=0.25, mask_zero=True))
        self.model.add(GRU(128))
        self.model.add(Dense(len(self.song_hash) + 1, activation=self.activation))

        self.model.compile(optimizer=self.optimizer,
                           loss=self.loss,
                           metrics=self.metrics)

        self.model.fit(self.x_train, self.y_train, nb_epoch=10, batch_size=10000)

        return self

    def evaluate(self):
        scores = self.model.evaluate(self.x_test, self.y_test)[1:]

        report = ""
        for metrics_name, score in zip(self.model.metrics_names[1:], scores):
            report += "%s: %.2f%%\n" % (metrics_name, score * 100)

        print(report)

        return self

    def get_predictions(self, seeds, top_k):
        preds = self.model(np.array([seeds]))[0]
        return preds.argsort()[-top_k:][::-1]
