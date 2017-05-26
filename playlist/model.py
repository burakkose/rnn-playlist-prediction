import numpy as np
from data import generate_data
from keras.layers import Dense, GRU, Embedding
from keras.models import Sequential
from keras.optimizers import Adam
from keras.preprocessing import sequence
from keras.utils.np_utils import to_categorical
from metrics import top_k_accuracy_func_list


class PlaylistGeneration:
    def __init__(self, path):
        self.optimizer = \
            Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
        self.model = Sequential()
        self.activation = 'softmax'
        self.loss = 'categorical_crossentropy'
        self.metrics = ['accuracy'].extend(top_k_accuracy_func_list([50, 100, 200, 300, 400, 500]))

        x, y, vocabulary, max_length = generate_data(path)

        self.vocabulary = vocabulary
        self.x_train = np.asarray(sequence.pad_sequences(x, maxlen=max_length), dtype="int64")
        self.y_train = to_categorical(y, None)  # It is not 0-indexed, use None

    def process(self):
        self.model.add(Embedding(len(self.y_train[0]), 50, dropout=0.25, mask_zero=True))
        self.model.add(GRU(1))
        self.model.add(Dense(len(self.y_train[0]), activation=self.activation))

        self.model.compile(optimizer=self.optimizer,
                           loss=self.loss,
                           metrics=self.metrics)

        self.model.fit(self.x_train, self.y_train, nb_epoch=1, batch_size=10000)

        return self

    def report(self):
        scores = self.model.evaluate(self.x_train, self.y_train)

        report = ""
        #for metrics_name, score in zip(self.model.metrics_names, scores):
        #    report += "%s: %.2f%%" % (metrics_name, scores * 100)

        #print(report)

        print(scores)

        return self

    def get_predictions(self, seeds):
        pass
