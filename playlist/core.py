from playlist.models.embedding_model_generator import EmbeddingModelGenerator
from playlist.tools.data import DatasetMode

EmbeddingModelGenerator(mode=DatasetMode.big).process().evaluate()