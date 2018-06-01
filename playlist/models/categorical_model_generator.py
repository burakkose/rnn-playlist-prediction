from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from keras.layers import *
from keras.models import Sequential
from keras.optimizers import Adam
from keras.preprocessing import sequence
from keras.utils.np_utils import to_categorical
from livelossplot import PlotLossesKeras

from playlist.config import *
from playlist.models.attention import Attention
from playlist.models.model_modes import *
from playlist.tools.data import DatasetMode, load
from playlist.tools.metrics import top_k_accuracy_func_list


class ModelGenerator:
    def __init__(self, mode=DatasetMode.small, model_name=ModelName.simple_gru):
        self.optimizer = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
        self.activation = 'softmax'
        self.loss = 'categorical_crossentropy'
        self.metrics = top_k_accuracy_func_list([50, 100, 200, 300, 400, 500])

        self.epochs = 15
        self.batch_size = 512
        self.validation_split = 0.1
        self.data_mode = mode

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

        self.train_len = len(x_train)
        self.test_len = len(x_test)

        self.validation_x, self.validation_y = [], []
        self.train_x, self.train_y = [], []

        if model_name is ModelName.simple_gru:
            self.weights_loc = TRAINED_MODELS_BASE_PATH + 'simple_gru_weights.best.hdf5'
            self.tb_logs_path = LOGS_BASE_PATH + 'simple_gru_logs'
            self.model = self.create_simple_gru_model()
        elif model_name is ModelName.bi_directional_lstm:
            self.weights_loc = TRAINED_MODELS_BASE_PATH + 'bi_lstm_weights.best.hdf5'
            self.tb_logs_path = LOGS_BASE_PATH + 'bi_lstm_logs'
            self.model = self.create_bi_lstm_model()
        elif model_name is ModelName.attention_bilstm:
            self.weights_loc = TRAINED_MODELS_BASE_PATH + 'ablstm_model_weights.best.hdf5'
            self.tb_logs_path = LOGS_BASE_PATH + 'ablstm_model_logs'
            self.model = self.create_ablstm_model()
        else:
            raise Exception("Unknown Model Name; ", model_name)

        self.callbacks = self._get_callback_functions()

    def create_simple_gru_model(self):
        model = Sequential()
        model.add(Embedding(len(self.song_hash) + 1, 50, mask_zero=True))
        model.add(SpatialDropout1D(rate=0.20))
        model.add(GRU(128))
        model.add(Dense(len(self.song_hash) + 1, activation=self.activation))

        model.compile(optimizer=self.optimizer, loss=self.loss, metrics=self.metrics)
        return model

    def create_bi_lstm_model(self):
        model = Sequential()
        model.add(Embedding(len(self.song_hash) + 1, 50, mask_zero=True))
        model.add(SpatialDropout1D(rate=0.20))
        model.add(Bidirectional(LSTM(128, dropout=0.2, recurrent_dropout=0.1)))
        model.add(Dense(len(self.song_hash) + 1, activation=self.activation))

        model.compile(optimizer=self.optimizer, loss=self.loss, metrics=self.metrics)
        return model

    def create_ablstm_model(self):
        model = Sequential()
        model.add(Embedding(len(self.song_hash) + 1, 50, mask_zero=True))
        model.add(SpatialDropout1D(rate=0.20))
        model.add(Bidirectional(LSTM(128, dropout=0.2, recurrent_dropout=0.1, return_sequences=True)))
        model.add(Attention(bias=False))
        model.add(Dense(60, activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(len(self.song_hash) + 1, activation=self.activation))
        model.compile(optimizer=self.optimizer, loss=self.loss, metrics=self.metrics)
        return model

    def evaluate(self):
        scores = self.model.evaluate_generator(generator=self._data_generator(mode=DataGeneratorMode.test),
                                               steps=self.test_len / self.batch_size + 1)[1:]

        report = ""
        for metrics_name, score in zip(self.model.metrics_names[1:], scores):
            report += "%s: %.2f%%\n" % (metrics_name, score * 100)

        print(report)

        return self

    def get_predictions(self, seeds, top_k):
        preds = self.model(np.array([seeds]))[0]
        return preds.argsort()[-top_k:][::-1]

    def _get_callback_functions(self):
        early_stopping = EarlyStopping(monitor='val_loss', patience=3)
        plot_losses = PlotLossesKeras()
        tb_callback = TensorBoard(log_dir=self.tb_logs_path, histogram_freq=0, write_graph=True,
                                  write_images=False, embeddings_freq=0, embeddings_layer_names=None,
                                  embeddings_metadata=None)

        checkpoint = ModelCheckpoint(self.weights_loc,
                                     monitor='val_loss',
                                     verbose=1,
                                     save_best_only=True,
                                     mode='min')
        return [early_stopping, plot_losses, tb_callback, checkpoint]

    def process(self):
        self.model.fit_generator(generator=self._data_generator(),
                                 validation_data=self._data_generator(mode=DataGeneratorMode.validation),
                                 validation_steps=len(self.validation_x) / self.batch_size + 1,
                                 epochs=self.epochs,
                                 verbose=1,
                                 steps_per_epoch=self.train_len / self.batch_size + 1,
                                 callbacks=self.callbacks)
        return self

    def _data_generator(self, mode=DataGeneratorMode.training):
        def gen_training_data():
            (x_train, y_train), (_, _), _ = load(self.data_mode)
            z = list(zip(x_train, y_train))
            pos = int(len(x_train) * self.validation_split)
            z_tr = z[pos:]
            z_val = z[:pos]
            self.validation_x, self.validation_y = zip(*z_val)
            self.train_x, self.train_y = zip(*z_tr)

            return x_train, y_train

        inputs, outputs = [], []

        if mode == DataGeneratorMode.training or mode == DataGeneratorMode.validation:
            if len(self.validation_x) == 0:
                inputs, outputs = gen_training_data()
            else:
                inputs, outputs = self.validation_x, self.validation_y
        else:
            (_, _), (x_test, y_test), _ = load(self.data_mode)
            inputs, outputs = x_test, y_test

        index = 0
        last_batch_x, last_batch_y = [], []

        while True:
            try:
                inp = inputs[index:index + self.batch_size]
                out = outputs[index:index + self.batch_size]

                last_batch_x = np.asarray(sequence.pad_sequences(inp, maxlen=self.max_length), dtype="int64")
                last_batch_y = to_categorical(out, len(self.song_hash) + 1)  # Zero is included
                index += self.batch_size
            except IndexError:
                index = 0
            except Exception as exp:
                logging.error('Data generator error', exp)
                index = 0
            yield last_batch_x, last_batch_y
