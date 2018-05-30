# rnn-playlist-prediction
Playlist prediction with recurrent neural networks on [Cornell Playlist Data Set](https://www.cs.cornell.edu/~shuochen/lme/data_page.html)

## Experiments with Categorical cross-entropy outputs
### On Small Data Set

| Model Name | func   | func_1 | func_2 | func_3 | func_4 | func_5 | Details |
|------------|--------|--------|--------|--------|--------|--------|---------|
| BiLSTM     | 30.16% | 43.52% | 60.52% | 71.56% | 78.38% | 82.85% | [Notebook](https://github.com/cenkcorapci/rnn-playlist-prediction/blob/master/bi-lstm.ipynb)        |
| SimpleGRU  |29.64%  | 42.83% | 59.58% | 70.79% | 77.60% | 82.07% | [Notebook](https://github.com/cenkcorapci/rnn-playlist-prediction/blob/master/simple-gru.ipynb)|
| LSTM_with_Attention  |31.42%  | 44.32% | 60.75% | 71.85% | 78.36% | 82.65% | [Notebook](https://github.com/cenkcorapci/rnn-playlist-prediction/blob/master/bi-lstm-with-attention.ipynb)|


### On Big Data Set



| Model Name | func   | func_1 | func_2 | func_3 | func_4 | func_5 | Details |
|------------|--------|--------|--------|--------|--------|--------|---------|
| LSTM_with_Attention  |20.74%  | 29.83% | 41.46% | 49.85% | 56.38% | 61.69% | [Notebook](https://github.com/cenkcorapci/rnn-playlist-prediction/blob/master/bi-lstm-with-attention.ipynb)|

## Experiments with Word2Vec embeddings as inputs and outputs

| Model Name | mae   | mse | acc | mape | cosine_proximity | Details |
|------------|--------|--------|--------|--------|--------|---------|
| SimpleGRU  |0.67  | 0.93 | 0.19 | 31.89 | -0.40  |[Notebook](https://github.com/cenkcorapci/rnn-playlist-prediction/blob/master/simple_gru_with_embedding.ipynb)|
