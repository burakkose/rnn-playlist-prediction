import logging
import os

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

LOGS_BASE_PATH = os.environ.get('LOGS_BASE_PATH', 'logs/')
TRAINED_MODELS_BASE_PATH = os.environ.get('TRAINED_MODELS_BASE_PATH', 'bin/trained_models/')
TRAINED_EMBEDDINGS_PATH = os.environ.get('TRAINED_EMBEDDINGS_PATH', 'bin/embeddings/embedding.txt')

EMBEDDING_DIM = 128
