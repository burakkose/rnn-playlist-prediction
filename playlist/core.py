from playlist.models.model_generator import ModelGenerator, ModelName

if __name__ == '__main__':
    ModelGenerator(model_name=ModelName.attention_bilstm).process().evaluate()
