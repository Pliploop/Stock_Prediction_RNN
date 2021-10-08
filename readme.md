
# Overview of data
 - reminder of objective of the project
 - Graphs showing data (open, close, high, low, volume)
 - explain what they represent
 - explain why the close drop in 2014
 - show data scaling is appropriate
 - show that the data is appropriate for prediction

# Recurrent neural networks

lecture material + explain the advantages and disadvantages of each method compared to vanilla

## RNN principles
## Improving upon RNNs
### LSTM
### GRU
### Bidirectional
### Stacked
### Memory and attention

# Architecture overview


  

# Converging on appropriate training values

 - Established baseline prediction with appropriate sequence length and batch size (show that accuracy decreases when long sequences and big batch sizes)
 - Tried multiple values for dropout, n_epochs, learning rate and hidden_size (amount of features taken into account (show accuracy decreases when dropout rises because model is already underfitting))
 - show that we tried predicting with only stock values and also volume values and using volume values proved best
 - show that bidirectional lstms are not a good idea here since they use the future, which is not possible when prediting stocks as opposed to translating for instance
 - show that layers were implemented and decreased accuracy
 - with final paramaters, compare GRU LSTM and RNN at same parameters


## Potential bonus ideas

 - Attention mechanism
 - time series feature extraction