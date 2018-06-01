# rnn-playlist-prediction
Playlist prediction with recurrent neural networks on [Cornell Playlist Data Set](https://www.cs.cornell.edu/~shuochen/lme/data_page.html)

## Experiments with Categorical cross-entropy outputs
### On Small Data Set

| Model Name | func   | func_1 | func_2 | func_3 | func_4 | func_5 | 
|------------|--------|--------|--------|--------|--------|--------|
| BiLSTM     | 30.16% | 43.52% | 60.52% | 71.56% | 78.38% | 82.85% |
| SimpleGRU  |29.64%  | 42.83% | 59.58% | 70.79% | 77.60% | 82.07% |
| BiLSTM_with_Attention  |31.42%  | 44.32% | 60.75% | 71.85% | 78.36% | 82.65% | 


### On Big Data Set
| Model Name | func   | func_1 | func_2 | func_3 | func_4 | func_5 |
|------------|--------|--------|--------|--------|--------|--------|
| SimpleGRU  |17.55%  | 26.16% | 37.59% | 45.82% | 52.10% | 57.11% |
| BiLSTM  |18.26%  | 27.16% | 39.05% | 47.47% | 53.92% | 59.24% | 
| BiLSTM_with_Attention  |20.74%  | 29.83% | 41.46% | 49.85% | 56.38% | 61.69% | 

### On Complete Data Set
*After 15 epochs*

| Model Name | func   | func_1 | func_2 | func_3 | func_4 | func_5 |
|------------|--------|--------|--------|--------|--------|--------|
| BiLSTM_with_Attention  |9.43%  | 14.51% | 19.64% | 22.18% | 23.74% | 25.69% | 

### Details
| Model Name | Notebook |
|------------|---------|
| BiLSTM     | [Notebook](https://github.com/cenkcorapci/rnn-playlist-prediction/blob/master/bi-lstm.ipynb)        |
| SimpleGRU  |[Notebook](https://github.com/cenkcorapci/rnn-playlist-prediction/blob/master/simple-gru.ipynb)|
| LSTM_with_Attention  |[Notebook](https://github.com/cenkcorapci/rnn-playlist-prediction/blob/master/bi-lstm-with-attention.ipynb)|



## Experiments with Word2Vec embeddings as inputs and outputs
| Model Name | mae   | mse | acc | mape | cosine_proximity | Details |
|------------|--------|--------|--------|--------|--------|---------|
| SimpleGRU  |0.67  | 0.93 | 0.19 | 31.89 | -0.40  |[Notebook](https://github.com/cenkcorapci/rnn-playlist-prediction/blob/master/simple_gru_with_embedding.ipynb)|
| BiLSTM  |0.66  | 0.87 | 0.16 | 243.73 | -0.41  |[Notebook](https://github.com/cenkcorapci/rnn-playlist-prediction/blob/master/bilstm_with_embedding.ipynb)|
