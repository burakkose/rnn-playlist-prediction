from playlist.models.model_generator import ModelGenerator, ModelName
from playlist.tools.data import DatasetMode

if __name__ == '__main__':
    ModelGenerator(mode=DatasetMode.big, model_name=ModelName.attention_bilstm).process().evaluate()
