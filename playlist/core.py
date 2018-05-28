from playlist.models.model_generator import ModelGenerator, ModelName

if __name__ == '__main__':
    ModelGenerator(model_name=ModelName.bi_directional_lstm).process().evaluate()
