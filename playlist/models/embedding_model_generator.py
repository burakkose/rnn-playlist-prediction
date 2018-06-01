from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from keras.layers import *
from keras.models import Sequential
from keras.optimizers import RMSprop
from keras.preprocessing.sequence import pad_sequences
from livelossplot import PlotLossesKeras

from playlist.config import *
from playlist.models.attention import Attention
from playlist.models.embeddings import SongEmbeddings
from playlist.models.model_modes import *
from playlist.tools.data import DatasetMode, load


class EmbeddingModelGenerator:
    def __init__(self, mode=DatasetMode.small, model_name=ModelName.simple_gru):
        self.optimizer = RMSprop(lr=0.001, rho=0.9, epsilon=None, decay=0.0)
        self.activation = 'tanh'
        self.loss = 'cosine'
        self.metrics = ['mae', 'mse', 'acc', 'mape', 'cosine', 'categorical_crossentropy']
        self.epochs = 50
        self.batch_size = 512
        self.validation_split = 0.1

        self.data_mode = mode
        v_index, embedding_matrix = SongEmbeddings().get_embeddings_matrix()
        self.embeddings_matrix = embedding_matrix
        self.vocab_index = v_index

        self.validation_x, self.validation_y = [], []
        self.train_x, self.train_y = [], []

        '''
        Index of songs in x_train(or test) starts from 1 because of zero padding.
        Index of songs in y_train(or test) starts from zero like song hash.
        For instance:
        In dataset, index of songA is 21.
        songA's index is 22 in x_train(or test)
        songA's index is 21 in y_train(or test).
        The goal is the neural network having the ability to ignore zero-paddings
        '''
        (x_train, y_train), (x_test, _), songs = load(mode)

        self.max_length = max([len(playlist) for playlist in x_train])
        self.song_hash = songs
        self.train_len = len(x_train)
        self.test_len = len(x_test)

        if model_name is ModelName.simple_gru:
            self.weights_loc = TRAINED_MODELS_BASE_PATH + 'e_simple_gru_weights.best.hdf5'
            self.tb_logs_path = LOGS_BASE_PATH + 'e_simple_gru_logs'
            self.model = self._create_simple_gru_model()
        elif model_name is ModelName.bi_directional_lstm:
            self.weights_loc = TRAINED_MODELS_BASE_PATH + 'e_bi_lstm_weights.best.hdf5'
            self.tb_logs_path = LOGS_BASE_PATH + 'e_bi_lstm_logs'
            self.model = self._create_bi_lstm_model()
        elif model_name is ModelName.attention_bilstm:
            self.weights_loc = TRAINED_MODELS_BASE_PATH + 'e_ablstm_model_weights.best.hdf5'
            self.tb_logs_path = LOGS_BASE_PATH + 'e_ablstm_model_logs'
            self.model = self._create_ablstm_model()
        else:
            raise Exception("Unknown Model Name; ", model_name)

        self.callbacks = self._get_callback_functions()

    def _create_simple_gru_model(self):
        model = Sequential()
        model.add(Embedding(input_dim=self.embeddings_matrix.shape[0],
                            output_dim=self.embeddings_matrix.shape[1],
                            weights=[self.embeddings_matrix],
                            input_length=self.max_length,
                            trainable=False))
        model.add(SpatialDropout1D(rate=0.20))
        model.add(GRU(EMBEDDING_DIM))
        model.add(Dense(EMBEDDING_DIM, activation=self.activation))

        model.compile(optimizer=self.optimizer, loss=self.loss, metrics=self.metrics)
        return model

    def _create_bi_lstm_model(self):
        model = Sequential()
        model.add(Embedding(input_dim=self.embeddings_matrix.shape[0],
                            output_dim=self.embeddings_matrix.shape[1],
                            weights=[self.embeddings_matrix],
                            input_length=self.max_length,
                            trainable=False))
        model.add(SpatialDropout1D(rate=0.20))
        model.add(Bidirectional(LSTM(EMBEDDING_DIM, dropout=0.2, recurrent_dropout=0.1)))
        model.add(Dense(EMBEDDING_DIM, activation=self.activation))

        model.compile(optimizer=self.optimizer, loss=self.loss, metrics=self.metrics)
        return model

    def _create_ablstm_model(self):
        model = Sequential()
        model.add(Embedding(input_dim=self.embeddings_matrix.shape[0],
                            output_dim=self.embeddings_matrix.shape[1],
                            weights=[self.embeddings_matrix],
                            input_length=self.max_length,
                            trainable=False))
        model.add(Bidirectional(LSTM(EMBEDDING_DIM, dropout=0.2, recurrent_dropout=0.1, return_sequences=True)))
        model.add(Attention(bias=False))
        model.add(Dense(EMBEDDING_DIM, activation=self.activation))
        model.compile(optimizer=self.optimizer, loss=self.loss, metrics=self.metrics)
        return model

    def process(self):
        self.model.fit_generator(generator=self._data_generator(),
                                 validation_data=self._data_generator(mode=DataGeneratorMode.validation),
                                 validation_steps=len(self.validation_x) / self.batch_size + 1,
                                 epochs=self.epochs,
                                 verbose=1,
                                 steps_per_epoch=self.train_len / self.batch_size + 1,
                                 callbacks=self.callbacks)
        return self

    def evaluate(self):
        scores = self.model.evaluate_generator(generator=self._data_generator(mode=DataGeneratorMode.test),
                                               steps=self.test_len / self.batch_size + 1)[1:]
        report = ""
        for metrics_name, score in zip(self.model.metrics_names[1:], scores):
            report += "%s: %.2f\n" % (metrics_name, score)

        print(report)

        return self

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
            def get_batch(i):
                try:
                    vectorized_inp = []
                    vectorized_out = []
                    for b in range(self.batch_size):
                        playlist = inputs[i + b]
                        vectorized_inp.append([self.vocab_index[str(song)] for song in playlist])
                        vectorized_out.append(self.embeddings_matrix[self.vocab_index[str(outputs[i + b])]])

                    x = np.asarray(
                        pad_sequences(vectorized_inp, maxlen=self.max_length))
                    y = np.asarray(vectorized_out)
                    return x, y, i
                except KeyError as key_error:
                    # logging.warning('Can\'t find key', key_error)
                    return get_batch(i + self.batch_size)

            try:
                last_batch_x, last_batch_y, index = get_batch(index)
                index += self.batch_size
            except IndexError:
                index = 0
            except Exception as exp:
                logging.error('Data generator error', exp)
                index = 0

            yield last_batch_x, last_batch_y
