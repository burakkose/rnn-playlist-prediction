import os.path

from gensim.models import Word2Vec, KeyedVectors

from playlist.config import *
from playlist.tools.data import *
import numpy as np


class SongEmbeddings:
    def get_playlists_as_sequences(self):
        (x_train, _), (x_test, _), _ = load(DatasetMode.big)

        c = np.concatenate((x_train, x_test), axis=0)
        c = [['$'] + [str(item) for item in r] for r in c]
        return np.array(c)

    def get_embeddings_matrix(self):
        if not os.path.exists(TRAINED_EMBEDDINGS_PATH):
            logging.info("Can't find pre-trained embeddings, training...")
            self._train()

        model = KeyedVectors.load_word2vec_format(TRAINED_EMBEDDINGS_PATH, binary=False)

        vocab_index = {}
        for i in range(len(model.vocab)):
            vocab_index[model.index2word[i]] = i

        embedding_matrix = np.zeros((len(model.vocab), EMBEDDING_DIM))
        for i in range(len(model.vocab)):
            embedding_vector = model[model.index2word[i]]
            if embedding_vector is not None:
                embedding_matrix[i] = embedding_vector

        return vocab_index, embedding_matrix

    def _train(self):
        seq = self.get_playlists_as_sequences()
        model = Word2Vec(seq, min_count=1, size=EMBEDDING_DIM)
        model.wv.save_word2vec_format(TRAINED_EMBEDDINGS_PATH, binary=False)
        logging.info('Word2Vec training is complete.')
