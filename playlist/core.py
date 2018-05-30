from playlist.models.embedding_model_generator import EmbeddingModelGenerator
from playlist.tools.data import DatasetMode
from playlist.models.model_modes import *

if __name__ == '__main__':
    EmbeddingModelGenerator(mode=DatasetMode.big, model_name=ModelName.attention_bilstm).process().evaluate()
